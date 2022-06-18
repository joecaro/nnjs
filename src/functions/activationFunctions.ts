import ActivationFunction from "../types/ActivationFunction";

const activationFunctions: activationFunctionsType = {
  relu: (value: number) => Math.max(0, value),
  sigmoid: (value: number) => 1 / (1 + Math.pow(Math.E, -value)),
  tanh: (value: number) => 2 / (1 + Math.pow(Math.E, -2 * value) + 1),
};

export const activationFunctionDerivs: activationFunctionsDerivativeType = {
  relu_d: (value: number) => {
    if (value >= 0) {
      return 1;
    } else {
      return 0;
    }
  },
  sigmoid_d: (x) => {
    let x1 = 1 / (1 + Math.exp(-x));
    return x1 * (1 - x1);
  },
  tanh_d: (value: number) => {
    let numer = Math.pow(Math.exp(2 * value) - 1, 2);
    let denom = Math.pow(Math.exp(2 * value) + 1, 2);
    return 1 - numer / denom;
  },
};

export type activationFunctionsType = {
  relu: ActivationFunction;
  sigmoid: ActivationFunction;
  tanh: ActivationFunction;
};

export type activationFunctionsDerivativeType = {
  relu_d: ActivationFunction;
  sigmoid_d: ActivationFunction;
  tanh_d: ActivationFunction;
};

export default activationFunctions;
