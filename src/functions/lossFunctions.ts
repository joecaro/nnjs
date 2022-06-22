import LossFunction from "../types/LossFunction";

export const lossFunctions: lossFunctionsType = {
  mse: (outputs, expectedValues) =>
    expectedValues.reduce((a, v, idx) => a + Math.pow(outputs[idx] - v, 2), 0) /
    expectedValues.length,
  mae: (outputs, expectedValues) =>
    expectedValues.reduce((a, v, idx) => a + Math.abs(outputs[idx] - v), 0) /
    expectedValues.length,
  lcl: (outputs, expectedValues) =>
    expectedValues.reduce(
      (a, v, idx) => a + Math.log(Math.cosh(outputs[idx] - v)),
      0
    ) / expectedValues.length,
};

export const lossFunctionDerivatives: LossFunctionDerivateType = {
  mse: (error: number) => {
    return 2 * error;
  },
  mae: (error: number) => {
    if (error >= 0) return 1;
    else return -1;
  },
  lcl: (error: number) => Math.tanh(error),
};

export type lossFunctionsType = {
  mse: LossFunction;
  mae: LossFunction;
  lcl: LossFunction;
};

export type LossFunctionDerivateType = {
  mse: (error: number) => number;
  mae: (error: number) => number;
  lcl: (error: number) => number;
};

export default lossFunctionsType;
