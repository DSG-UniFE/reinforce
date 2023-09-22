# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'reinforce/q_function_ann'

describe Reinforce::QFunctionANN do
  let(:q_function) { Reinforce::QFunctionANN.new(2, 2, 0.01, 0.99) }
  let(:q_target_function) { Reinforce::QFunctionANN.new(2, 2, 0.01, 0.99) }

  it 'can be soft updated from another ANN-based Q function' do
    tau = 0.9
    q_copied_params = q_function.parameters.map(&:clone)
    qtar_copied_params = q_target_function.parameters.map(&:clone)
    q_target_function.soft_update(q_function, tau)
    q_target_function.parameters.each_with_index do |p, i|
      p.flatten.to_a do |value|
        expect(value).to be_within(1E-3).of(qtar_copied_params[i] * tau + q_copied_params[i] * (1.0 - tau))
      end
    end
  end
end
