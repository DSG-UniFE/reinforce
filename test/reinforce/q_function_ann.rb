# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'reinforce/q_function_ann'

describe Reinforce::QFunctionANN do
  let(:q_function) { Reinforce::QFunctionANN.new(2, 2, 0.01, 0.99) }
  let(:q_target_function) { Reinforce::QFunctionANN.new(2, 2, 0.01, 0.99) }

  it 'can be soft updated from another ANN-based Q function' do
    # Regression test for two pre-existing bugs found while extracting
    # Reinforce::Networks.soft_update!:
    #   1. `p.flatten.to_a do |value| ... end` never actually ran the
    #      block below it (Tensor#to_a does not yield), so this test
    #      passed regardless of whether soft_update was correct.
    #   2. The expected value was computed as
    #      target_old * tau + online * (1 - tau), which is the reverse of
    #      what #soft_update actually (and correctly) computes:
    #      target_old * (1 - tau) + online * tau -- see
    #      Reinforce::Networks.soft_update! for the convention this
    #      follows (tau is how much of `online` is blended in).
    tau = 0.9
    q_copied_params = q_function.parameters.map(&:clone)
    qtar_copied_params = q_target_function.parameters.map(&:clone)
    q_target_function.soft_update(q_function, tau)
    q_target_function.parameters.each_with_index do |p, i|
      expected = (qtar_copied_params[i] * (1.0 - tau)) + (q_copied_params[i] * tau)
      p.flatten.to_a.zip(expected.flatten.to_a).each do |value, expected_value|
        expect(value).to be_within(1E-3).of(expected_value)
      end
    end
  end
end
