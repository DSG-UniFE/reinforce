# frozen_string_literal: true

require "reinforce"

describe Reinforce::Metrics::Recorder do
  let(:recorder) { subject.new }

  it "reads back recorded values per key, oldest first" do
    recorder.record(:loss, 0.5)
    recorder.record(:loss, 0.25)
    recorder.record(:episode_reward, 3)

    expect(recorder[:loss]).to be == [0.5, 0.25]
    expect(recorder[:episode_reward]).to be == [3]
    expect(recorder.keys).to be == [:loss, :episode_reward]
  end

  it "returns an empty series for keys that were never recorded" do
    expect(recorder[:loss]).to be == []
    expect(recorder.key?(:loss)).to be == false
    expect(recorder.last(:loss)).to be_nil
    expect(recorder.mean(:loss)).to be_nil
  end

  it "accepts string keys as their symbol equivalents" do
    recorder.record("loss", 1.0)

    expect(recorder[:loss]).to be == [1.0]
    expect(recorder["loss"]).to be == [1.0]
  end

  it "does not let callers alter the recorded history" do
    recorder.record(:loss, 1.0)
    recorder[:loss] << 2.0
    recorder.to_h[:loss] << 3.0
    recorder.each { |_key, values| values << 4.0 }

    expect(recorder[:loss]).to be == [1.0]
  end

  it "computes the last value and the (windowed) mean" do
    [1, 2, 3, 4].each { |value| recorder.record(:episode_reward, value) }

    expect(recorder.last(:episode_reward)).to be == 4
    expect(recorder.mean(:episode_reward)).to be_within(1e-12).of(2.5)
    expect(recorder.mean(:episode_reward, window: 2)).to be_within(1e-12).of(3.5)
  end

  it "forwards every value to its sinks, with the step defaulting to the per-key index" do
    received = []
    recorder.add_sink(->(key, value, step) { received << [key, value, step] })

    recorder.record(:loss, 0.5)
    recorder.record(:episode_reward, 3)
    recorder.record(:loss, 0.25)
    recorder.record(:loss, 0.125, step: 100)

    expect(received).to be == [[:loss, 0.5, 0], [:episode_reward, 3, 0], [:loss, 0.25, 1], [:loss, 0.125, 100]]
  end

  it "records several metrics at once with a shared step" do
    received = []
    recorder.add_sink(->(key, value, step) { received << [key, value, step] })

    recorder.record_all({q_loss: 1.0, policy_loss: 2.0}, step: 7)

    expect(recorder.to_h).to be == {q_loss: [1.0], policy_loss: [2.0]}
    expect(received).to be == [[:q_loss, 1.0, 7], [:policy_loss, 2.0, 7]]
  end

  it "accepts sinks at construction time" do
    received = []
    recorder = subject.new(sinks: [->(key, value, _step) { received << [key, value] }])

    recorder.record(:loss, 1.0)

    expect(received).to be == [[:loss, 1.0]]
  end

  it "rejects sinks that do not respond to #call" do
    expect { recorder.add_sink(Object.new) }.to raise_exception(ArgumentError)
  end
end
