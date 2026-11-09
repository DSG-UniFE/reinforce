# frozen_string_literal: true

require "stringio"
require "reinforce"

describe Reinforce::Metrics::ConsoleSink do
  let(:output) { StringIO.new }

  it "prints the mean of every metric since the last line, once per `every` trigger values" do
    recorder = Reinforce::Metrics::Recorder.new(sinks: [subject.new(output:, every: 2)])

    recorder.record(:loss, 1.0)
    recorder.record(:episode_reward, 2)
    recorder.record(:loss, 3.0)
    recorder.record(:episode_reward, 4)
    recorder.record(:episode_length, 10)
    recorder.record(:episode_reward, 6)
    recorder.record(:episode_reward, 8)

    expect(output.string.lines).to be == [
      "[episode_reward 2] episode_reward=3 loss=2\n",
      "[episode_reward 4] episode_reward=7 episode_length=10\n"
    ]
  end

  it "can be triggered by any metric and formats values with the given precision" do
    sink = subject.new(output:, every: 1, trigger: :loss, precision: 2)

    sink.call(:loss, 0.123456, 0)

    expect(output.string).to be == "[loss 1] loss=0.12\n"
  end

  it "prints nothing until the trigger metric is recorded `every` times" do
    sink = subject.new(output:, every: 3)

    sink.call(:loss, 1.0, 0)
    sink.call(:episode_reward, 1.0, 0)
    sink.call(:episode_reward, 1.0, 1)

    expect(output.string).to be == ""
  end

  it "rejects a non-positive `every`" do
    expect { subject.new(every: 0) }.to raise_exception(ArgumentError)
  end
end
