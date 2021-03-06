import ActivationFunction from "../types/ActivationFunction";

const activationFunctions: activationFunctionsType = {
  relu: (value: number) => Math.max(0, value),
  sigmoid: (value: number) => 1 / (1 + Math.pow(Math.E, -value)),
  tanh: (value: number) => 2 / (1 + Math.pow(Math.E, -2 * value) + 1),
};

export const activationFunctionStrings = {
  relu: "(value) => Math.max(0, value)",
  sigmoid: "(value) => 1 / (1 + Math.pow(Math.E, -value))",
  tanh: "(value) => 2 / (1 + Math.pow(Math.E, -2 * value) + 1)",
};

export const activationFunctionDerivatives: activationFunctionsDerivativeType =
  {
    relu: (value: number) => {
      if (value >= 0) {
        return 1;
      } else {
        return 0;
      }
    },
    sigmoid: (x) => {
      // let x1 = 1 / (1 + Math.pow(Math.E, -x));
      return x * (1 - x);
    },
    tanh: (value: number) => {
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
  relu: ActivationFunction;
  sigmoid: ActivationFunction;
  tanh: ActivationFunction;
};

export default activationFunctions;
