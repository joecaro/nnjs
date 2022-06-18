import LossFunction from "../types/LossFunction";

export const lossFunctions: lossFunctionsType = {
  mse: (outputs, expectedValues) =>
    expectedValues.reduce((a, v, idx) => a + Math.pow(outputs[idx] - v, 2), 0) /
    expectedValues.length,
  mae: (outputs, expectedValues) =>
    expectedValues.reduce((a, v, idx) => a + Math.abs(outputs[idx] - v), 0) /
    expectedValues.length,
};

export const lossFunctions_D: LossFunctionType_D = {
  mse: (output: number, expectedValue: number) => {
    return 2 * (output - expectedValue);
  },
  mae: (output: number, expectedValue: number) => {
    if (output > expectedValue) return 1;
    else return -1;
  },
};

export type lossFunctionsType = {
  mse: LossFunction;
  mae: LossFunction;
};

export type LossFunctionType_D = {
  mse: (output: number, expectedValue: number) => number;
  mae: (output: number, expectedValue: number) => number;
};

export default lossFunctionsType;
