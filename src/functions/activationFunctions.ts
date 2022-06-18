import ActivationFunction from "../types/ActivationFunction";

const activationFunctions: activationFunctionsType = {
  relu: (value: number) => Math.max(0, value),
  relu_d: (value: number) => {
    if (value >= 0) {
      return 1;
    } else {
      return 0;
    }
  },
  sigmoid: (value: number) => 1 / (1 + Math.pow(Math.E, -value)),
  sigmoid_d: (x) => {
    let x1 = 1 / (1 + Math.exp(-x));
    return x1 * (1 - x1);
  },
  tanh: (value: number) => 2 / (1 + Math.pow(Math.E, -2 * value) + 1),
  tanh_d: (value: number) => {
    let numer = Math.pow(Math.exp(2 * value) - 1, 2);
    let denom = Math.pow(Math.exp(2 * value) + 1, 2);
    return 1 - numer / denom;
  },
};

export type activationFunctionsType = {
  relu: ActivationFunction;
  relu_d: ActivationFunction;
  sigmoid: ActivationFunction;
  sigmoid_d: ActivationFunction;
  tanh: ActivationFunction;
  tanh_d: ActivationFunction;
};

export default activationFunctions;
