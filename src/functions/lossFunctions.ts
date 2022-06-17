import LossFunction from "../types/LossFunction";

const lossFunctions: lossFunctionsType = {
  mse: (outputs, expectedValues) =>
    expectedValues.reduce((a, v, idx) => a + Math.pow(v - outputs[idx], 2), 0) /
    expectedValues.length,
  mae: (outputs, expectedValues) =>
    expectedValues.reduce((a, v, idx) => a + Math.abs(v - outputs[idx]), 0) /
    expectedValues.length,
};

export type lossFunctionsType = {
  mse: LossFunction;
  mae: LossFunction;
};

export default lossFunctions;
