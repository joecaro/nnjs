import ActivationFunction from "../types/ActivationFunction";

const activationFunctions: activationFunctionsType = {
  relu: (value: number) => Math.max(0, value),
  sigmoid: (value: number) => 1 / (1 + Math.pow(Math.E, -value)),
  tanh: (value: number) => 2 / (1 + Math.pow(Math.E, -2 * value) + 1),
};

export type activationFunctionsType = {
  relu: ActivationFunction;
  sigmoid: ActivationFunction;
  tanh: ActivationFunction;
};

export default activationFunctions;
