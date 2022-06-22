type LossFunction = (outputs: number[], expectedValues: number[]) => number;

export type LossFunctionDerivative = (error: number) => number;

export default LossFunction;
