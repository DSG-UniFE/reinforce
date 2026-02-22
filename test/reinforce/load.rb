# frozen_string_literal: true

require 'open3'

describe 'require reinforce' do
  it 'does not print to stdout and does not fail when torch is unavailable' do
    stdout, stderr, status = Open3.capture3(ENV, 'ruby', '-Ilib', '-e', 'require "reinforce"')

    expect(status.success?).to be == true
    expect(stdout).to be == ''
    expect(stderr).to be == ''
  end
end
