type LossFunction = (outputs: number[], expectedValues: number[]) => number;

export type LossFunctionDerivative = (
  output: number,
  expectedValue: number
) => number;

export default LossFunction;
