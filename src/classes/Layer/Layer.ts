import activationFunctions, {
  activationFunctionDerivatives,
  activationFunctionsType,
} from "../../functions/activationFunctions";
import Matrix from "../Matrix/Matrix";
import LayerTypes from "../../types/LayerTypes";

export default class Layer {
  numberOfNodes: number;
  numberOfInputs: number;
  type: string;
  weights: Matrix;
  biases: Matrix;
  activationFunction: keyof activationFunctionsType;

  constructor(
    numberOfNodes: number,
    numberOfInputs: number,
    type: keyof LayerTypes,
    activationFunction: keyof activationFunctionsType
  ) {
    this.numberOfNodes = numberOfNodes;
    this.numberOfInputs = numberOfInputs;
    this.type = type;

    this.weights = new Matrix(numberOfNodes, numberOfInputs);
    this.biases = new Matrix(numberOfNodes, 1);

    this.activationFunction = activationFunction;
  }

  randomize() {
    this.weights.randomize();
    this.biases.randomize();
  }

  generateOutputs(inputs: Matrix) {
    let outputs = Matrix.multiply(this.weights, inputs);
    outputs.add(this.biases);
    outputs.map(activationFunctions[this.activationFunction]);

    return outputs;
  }

  generateGradients(outputs: Matrix) {
    let gradients = Matrix.map(
      outputs,
      activationFunctionDerivatives[this.activationFunction]
    );
    return gradients;
  }

  getWeights() {
    return this.weights;
  }

  updateWeights(adjustments: Matrix) {
    this.weights.add(adjustments);
  }

  updateBiases(adjustments: Matrix) {
    this.biases.add(adjustments);
  }

  updateInputs(numberOfInputs: number) {
    this.numberOfInputs = numberOfInputs;

    this.weights = new Matrix(this.numberOfNodes, numberOfInputs);
  }

  getWeightLog() {
    return this.weights.getLog();
  }
}
