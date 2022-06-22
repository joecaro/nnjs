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
  mse: (output: number, expectedValue: number) => {
    return 2 * (output - expectedValue);
  },
  mae: (output: number, expectedValue: number) => {
    if (output > expectedValue) return 1;
    else return -1;
  },
  lcl: (output: number, expectedValue: number) =>
    Math.tanh(output - expectedValue),
};

export type lossFunctionsType = {
  mse: LossFunction;
  mae: LossFunction;
  lcl: LossFunction;
};

export type LossFunctionDerivateType = {
  mse: (output: number, expectedValue: number) => number;
  mae: (output: number, expectedValue: number) => number;
  lcl: (output: number, expectedValue: number) => number;
};

export default lossFunctionsType;
