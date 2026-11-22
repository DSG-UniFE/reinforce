# frozen_string_literal: true

require "ltn"
require_relative "bounded_sum"
require_relative "../environments/multi_cluster"

module Reinforce
  module Shields
    # The safety of replica allocations in Reinforce::Environments::MultiCluster,
    # as a differentiable score in [0, 1] computed by a Logic Tensor Network
    # from probabilistic sensors.
    #
    # **Sensors.** The free amount of each resource is uncertain, and the
    # sensors give the probability that a cluster has at least the amount
    # its quota of replicas needs. How depends on the environment's
    # telemetry (see docs/sensor-model.md):
    #
    # - *legacy* (max_delay or telemetry_interval): telemetry may be up to D
    #   steps old, so after d steps the free amount lies between the
    #   measurement minus d times the most a step can consume and the
    #   measurement plus d times the most a step can recover (within the
    #   capacity). Taking it uniform in that interval, and averaging over d
    #   in 0..D, gives the probability (an indicator when the interval is
    #   empty, e.g. D = 0).
    # - *telemetry* (MultiCluster::Telemetry): the observation gives the
    #   free amount y the agent assumes, the age a of its reading, the other
    #   tenants' estimated drift v and volatility s, and the probability q
    #   that they change nothing over the age scale A. Over the age, they
    #   change nothing with probability p0 = q^(a / A), and then the free
    #   amount is exactly y; otherwise their consumption grows by a normal
    #   amount whose mean and variance make the mixture's v a and s^2 a.
    #   The probability of at least x, for x in (0, capacity], is then
    #   p0 [y >= x] + (1 - p0) (1 - Phi((x - y + m) / sd)), with m and sd
    #   the normal's (1 below 0 and 0 above the capacity), with p0, m and sd
    #   per resource (other tenants' services need e.g. GPUs only now and
    #   then).
    #
    # From these, per cluster:
    #
    # - `fit` (per resource): the cluster has enough of the resource.
    # - `comfortable`: the allocation would leave at least 1 - comfort_fraction
    #   of the available resources free, given that it fits.
    # - `reject_safe` (per request): rejection is justified, i.e. the fraction
    #   of the rejected replicas that the clusters' estimated capacity (by
    #   bisection on the fit probability) could not host anyway.
    #
    # **Knowledge.** The allocation sends a replica to cluster c with
    # probability `chosen(c)` (its share), or rejects it with the reject
    # share. It is safe if the chosen cluster fits in every resource and is
    # not tight (fits, but not comfortably, which is penalized with
    # probability tight_penalty), or if rejection is chosen and justified:
    #
    #   safe = (exists c: chosen(c) and full_fit(c) and not (tight_penalty and not comfortable(c)))
    #          or (chosen(reject) and reject_safe)
    #   full_fit(c) = cpu_fit(c) and memory_fit(c) and gpu_fit(c) and bandwidth_fit(c)
    #
    # with the product t-norm, the standard negation, and Łukasiewicz's
    # disjunction and existential quantifier (BoundedSum). Since the sensor
    # events are taken as independent and the destinations are mutually
    # exclusive, this score is exactly the probability that the allocation
    # is safe under the corresponding probabilistic logic program.
    class MultiClusterSafety
      # Quotas (in replicas) below this count as zero.
      QUOTA_RESOLUTION = 1e-4

      # Fit probabilities below this count as zero.
      TINY = 1e-12

      # @parameter environment [Reinforce::Environments::MultiCluster] the
      #   environment whose observations are scored; only its configuration
      #   is used.
      # @parameter comfort_fraction [Float] the fraction of the available
      #   resources a comfortable allocation uses at most.
      # @parameter tight_penalty [Float] the probability that a tight
      #   allocation is unsafe.
      # @parameter capacity_threshold [Float] the fit probability at which a
      #   cluster's capacity is estimated.
      # @parameter bisection_steps [Integer] the bisection steps of that estimate.
      def initialize(environment, comfort_fraction: 0.8, tight_penalty: 0.3, capacity_threshold: 0.5, bisection_steps: 6)
        @layout = environment.observation_layout
        @clusters = environment.clusters
        config = environment.config
        resources = Reinforce::Environments::MultiCluster::RESOURCES
        @telemetry = config.telemetry
        @max_delay = environment.telemetry_delay_bound
        @replica_limit = config.replica_limit
        max_demand = config.max_demand_per_resource
        @capacity = Torch.tensor(environment.capacities)
        @max_demand = Torch.tensor(resources.map { |resource| max_demand[resource].to_f })
        # The most a step can consume (a whole request) and give back (see
        # MultiCluster#release_bound).
        @max_consumption = @max_demand * @replica_limit
        release_bound = environment.release_bound
        @max_recovery = Torch.tensor(resources.map { |resource| release_bound.fetch(resource).to_f })
        @comfort_fraction = comfort_fraction
        @tight_penalty = tight_penalty
        @capacity_threshold = capacity_threshold
        @bisection_steps = bisection_steps
        build_formula
      end

      # @parameter observations [Torch::Tensor] batch x observation size.
      # @parameter shares [Torch::Tensor] batch x (clusters + 1) allocation
      #   shares (non-negative, summing to 1), the last one for rejection.
      # @returns [Torch::Tensor] the safety score of each allocation (batch).
      def score(observations, shares)
        evaluate(**sensors(observations, shares))
      end

      # The truth value of the safety formula for given sensor readings.
      #
      # @parameter chosen [Torch::Tensor] batch x clusters: the cluster shares.
      # @parameter fit [Torch::Tensor] batch x clusters x resources.
      # @parameter comfortable [Torch::Tensor] batch x clusters.
      # @parameter reject_chosen [Torch::Tensor] batch: the reject shares.
      # @parameter reject_safe [Torch::Tensor] batch.
      # @returns [Torch::Tensor] batch.
      def evaluate(chosen:, fit:, comfortable:, reject_chosen:, reject_safe:)
        batch_size, clusters = chosen.shape
        tables = {chosen: chosen, comfortable: comfortable, reject_chosen: reject_chosen, reject_safe: reject_safe,
                  fit: fit.unbind(2), tight_penalty: Torch.full([batch_size], @tight_penalty, dtype: chosen.dtype)}
        state = LTN::Variable.new("s", Torch.arange(batch_size, dtype: :float32))
        cluster = LTN::Variable.new("c", Torch.arange(clusters, dtype: :float32))

        fits = @fit.map { |predicate| predicate.call(state, cluster, tables: tables) }
        full_fit = @and.call(@and.call(fits[0], fits[1]), @and.call(fits[2], fits[3]))
        tight = @and.call(@tight_penalty_rule.call(state, tables: tables), @not.call(@comfortable.call(state, cluster, tables: tables)))
        safe_cluster = @and.call(@chosen.call(state, cluster, tables: tables), @and.call(full_fit, @not.call(tight)))
        safe_rejection = @and.call(@reject_chosen.call(state, tables: tables), @reject_safe.call(state, tables: tables))
        @or.call(@exists.call(cluster, safe_cluster), safe_rejection).value
      end

      # The sensor readings for allocations (see the class comment).
      #
      # @returns [Hash{Symbol => Torch::Tensor}] the keyword arguments of #evaluate.
      def sensors(observations, shares)
        free, replicas, demand = decode(observations)
        @uncertainty = uncertainty(observations) if @telemetry
        cluster_shares = shares.narrow(1, 0, @clusters)
        reject_share = shares.select(1, @clusters)
        amounts = (cluster_shares * replicas.unsqueeze(1)).unsqueeze(2) * demand.unsqueeze(1)

        fit = availability(free, amounts)
        full_fit = fit.prod(2)
        comfortable_fit = availability(free, amounts / @comfort_fraction).prod(2)
        fits = full_fit.gt(TINY)
        comfortable = Torch.where(fits, Torch.clamp(comfortable_fit / Torch.where(fits, full_fit, Torch.ones_like(full_fit)), 0.0, 1.0),
          Torch.zeros_like(full_fit))

        unavoidable = Torch.clamp(replicas - capacity_estimate(free, demand, replicas).sum(1), min: 0.0)
        rejected = reject_share * replicas
        rejects = rejected.gt(QUOTA_RESOLUTION)
        reject_safe = Torch.where(rejects, Torch.clamp_max(unavoidable / Torch.where(rejects, rejected, Torch.ones_like(rejected)), 1.0),
          Torch.ones_like(rejected))

        {chosen: cluster_shares, fit: fit, comfortable: comfortable, reject_chosen: reject_share, reject_safe: reject_safe}
      end

      # The probability that each cluster accommodates the replicas the
      # shares give it (fit in every resource), for calibration (see
      # Shields::Calibration).
      #
      # @returns [Torch::Tensor] batch x clusters.
      def fit_probabilities(observations, shares)
        sensors(observations, shares)[:fit].prod(2)
      end

      # How far the allocated replicas are from the request's dependency: the
      # mean normalized latency over the cluster shares, 1 if everything is
      # rejected for a request with a dependency, and 0 without one.
      #
      # @returns [Torch::Tensor] batch.
      def latency(observations, shares)
        latency = observations.narrow(1, @layout.latency, @clusters).to(shares.dtype)
        dependent = observations.narrow(1, @layout.dependency, @clusters).sum(1).gt(0.5)
        cluster_shares = shares.narrow(1, 0, @clusters)
        allocated = cluster_shares.sum(1)
        some = allocated.gt(TINY)
        mean = (cluster_shares * latency).sum(1) / Torch.where(some, allocated, Torch.ones_like(allocated))
        Torch.where(some, mean, Torch.where(dependent, Torch.ones_like(mean), Torch.zeros_like(mean)))
      end

      private

      def build_formula
        @and = LTN::Connective.new(LTN::FuzzyOps::AndProd.new(stable: false))
        @not = LTN::Connective.new(LTN::FuzzyOps::NotStandard.new)
        @or = LTN::Connective.new(LTN::FuzzyOps::OrLuk.new)
        @exists = LTN::Quantifier.new(BoundedSum.new, :exists)

        @fit = 4.times.map { |resource| cluster_predicate { |tables| tables[:fit][resource] } }
        @comfortable = cluster_predicate { |tables| tables[:comfortable] }
        @chosen = cluster_predicate { |tables| tables[:chosen] }
        @reject_chosen = state_predicate(:reject_chosen)
        @reject_safe = state_predicate(:reject_safe)
        @tight_penalty_rule = state_predicate(:tight_penalty)
      end

      # A predicate on (state, cluster) that looks its truth values up in a
      # batch x clusters table.
      def cluster_predicate(&table)
        LTN::Predicate.new do |state, cluster, tables:|
          values = table.call(tables)
          indices = (state.reshape(-1).long * values.shape[1]) + cluster.reshape(-1).long
          values.reshape(-1).index_select(0, indices)
        end
      end

      # A predicate on states that looks its truth values up in a batch table.
      def state_predicate(key)
        LTN::Predicate.new do |state, tables:|
          tables[key].index_select(0, state.reshape(-1).long)
        end
      end

      # @returns [Array(Torch::Tensor)] the measured free resources (batch x
      #   clusters x resources; with telemetry, the ones the agent assumes),
      #   the replica counts (batch) and the per-replica demands (batch x
      #   resources), in physical units.
      def decode(observations)
        resources = @capacity.size(0)
        free = observations.narrow(1, @layout.free, @clusters * resources).reshape(-1, @clusters, resources) * @capacity
        replicas = Torch.round(observations.select(1, @layout.replicas) * @replica_limit)
        demand = Torch.round(observations.narrow(1, @layout.demand, resources) * @max_demand)
        [free, replicas, demand]
      end

      # With telemetry: the age of each cluster's reading (batch x clusters).
      def age(observations)
        observations.narrow(1, @layout.age, @clusters) * @telemetry.age_scale
      end

      # With telemetry: the other tenants' estimated consumption per time
      # unit (batch x clusters x resources).
      def drift(observations)
        resources = @capacity.size(0)
        normalized = observations.narrow(1, @layout.drift, @clusters * resources).reshape(-1, @clusters, resources)
        ((normalized * 2.0) - 1.0) * @capacity / @telemetry.age_scale
      end

      # With telemetry: the other tenants' estimated volatility per square
      # root of a time unit (batch x clusters x resources).
      def volatility(observations)
        resources = @capacity.size(0)
        normalized = observations.narrow(1, @layout.volatility, @clusters * resources).reshape(-1, @clusters, resources)
        normalized * @capacity / Math.sqrt(@telemetry.age_scale)
      end

      # With telemetry: over each reading's age, the probability that the
      # other tenants change nothing (`quiet`) and,
      # if they do, the mean growth of their consumption and its standard
      # deviation, matching the drift and volatility of the mixture (batch x
      # clusters x resources each).
      def uncertainty(observations)
        age = age(observations).unsqueeze(2)
        scale = @telemetry.age_scale
        resources = @capacity.size(0)
        quiet = observations.narrow(1, @layout.quiet, @clusters * resources).reshape(-1, @clusters, resources).clamp(TINY, 1.0)**(age / scale)
        active = Torch.clamp(1.0 - quiet, min: TINY)
        mean = drift(observations) * age
        variance = (volatility(observations)**2) * age
        shift = mean / active
        spread = Torch.sqrt(Torch.clamp(((variance + (mean**2)) / active) - (shift**2), min: 0.0))
        {quiet: quiet, shift: shift, spread: spread}
      end

      # The probability that the free amount of each resource is at least
      # `amounts`: with telemetry, from the mixture of the last #sensors
      # call around the assumed `free` amounts; otherwise averaged over the
      # possible delays of the measurement.
      def availability(free, amounts)
        return mixture_availability(free, amounts) if @telemetry

        total = Torch.zeros_like(amounts)
        (0..@max_delay).each do |delay|
          low = Torch.clamp(free - (@max_consumption * delay), min: 0.0)
          high = Torch.minimum(@capacity.expand_as(free), free + (@max_recovery * delay))
          spread = high.gt(low)
          width = Torch.where(spread, high - low, Torch.ones_like(high))
          total += Torch.where(spread, Torch.clamp((high - amounts) / width, 0.0, 1.0), free.ge(amounts).to(amounts.dtype))
        end
        total / (@max_delay + 1)
      end

      # Per resource, the marginal probability of the mixture.
      def mixture_availability(assumed, amounts)
        unchanged, changed = mixture_parts(assumed, amounts)
        quiet = @uncertainty[:quiet].expand_as(amounts)
        (quiet * unchanged) + ((1.0 - quiet) * changed)
      end

      # Per resource, whether the assumed amount fits `amounts` if the other
      # tenants changed nothing, and the probability that it does if they
      # did, censored to [0, capacity].
      def mixture_parts(assumed, amounts)
        capacity = @capacity.expand_as(amounts)
        mean = assumed - @uncertainty[:shift].expand_as(amounts)
        spread = @uncertainty[:spread].expand_as(amounts)
        uncertain = spread.gt(TINY)
        safe_spread = Torch.where(uncertain, spread, Torch.ones_like(spread))
        changed = Torch.where(uncertain, 0.5 * Torch.erfc((amounts - mean) / (safe_spread * Math.sqrt(2.0))), mean.ge(amounts).to(amounts.dtype))
        unchanged = assumed.ge(amounts).to(amounts.dtype)
        [unchanged, changed].map do |part|
          Torch.where(amounts.le(0.0), Torch.ones_like(amounts), Torch.where(amounts.gt(capacity), Torch.zeros_like(amounts), part))
        end
      end

      # The most replicas each cluster hosts with fit probability at least
      # capacity_threshold, estimated by bisection over [0, replicas]: the
      # lower end of the final interval.
      def capacity_estimate(free, demand, replicas)
        Torch.no_grad do
          low = Torch.zeros_like(free.select(2, 0))
          high = replicas.unsqueeze(1).expand_as(low).clone
          @bisection_steps.times do
            middle = (low + high) / 2.0
            fits = availability(free, middle.unsqueeze(2) * demand.unsqueeze(1)).prod(2).ge(@capacity_threshold)
            low = Torch.where(fits, middle, low)
            high = Torch.where(fits, high, middle)
          end
          low
        end
      end
    end
  end
end
