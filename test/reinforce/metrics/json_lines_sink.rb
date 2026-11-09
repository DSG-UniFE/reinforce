# frozen_string_literal: true

require "json"
require "stringio"
require "tmpdir"
require "reinforce"

describe Reinforce::Metrics::JSONLinesSink do
  let(:clock) { -> { 1_700_000_000.5 } }

  it "writes one JSON object per recorded value" do
    io = StringIO.new
    recorder = Reinforce::Metrics::Recorder.new(sinks: [subject.new(io, clock:)])

    recorder.record(:loss, 0.5)
    recorder.record(:episode_reward, 3, step: 9)

    records = io.string.lines.map { |line| JSON.parse(line) }
    expect(records).to be == [
      {"key" => "loss", "value" => 0.5, "step" => 0, "time" => 1_700_000_000.5},
      {"key" => "episode_reward", "value" => 3, "step" => 9, "time" => 1_700_000_000.5}
    ]
  end

  it "keeps non-finite values instead of failing on them" do
    io = StringIO.new
    sink = subject.new(io, clock:)

    sink.call(:loss, Float::NAN, 0)
    sink.call(:loss, Float::INFINITY, 1)

    values = io.string.lines.map { |line| JSON.parse(line, allow_nan: true)["value"] }
    expect(values[0].nan?).to be == true
    expect(values[1].infinite?).to be == 1
  end

  it "opens, writes and closes a file when given a path" do
    Dir.mktmpdir do |directory|
      path = File.join(directory, "metrics.jsonl")
      sink = subject.new(path, clock:)

      sink.call(:loss, 1.0, 0)
      sink.close
      sink.close # idempotent

      expect(JSON.parse(File.read(path))["value"]).to be_within(1e-12).of(1.0)
    end
  end

  it "does not close an IO it was handed" do
    io = StringIO.new
    subject.new(io, clock:).close

    expect(io.closed?).to be == false
  end
end
