import ActivationFunction from "../../types/ActivationFunction";
import activationFunctions, {
  activationFunctionDerivatives,
  activationFunctionsType,
} from "../../functions/activationFunctions";
import Matrix from "../Matrix/Matrix";

export default class Layer {
  numberOfNodes: number;
  numberOfInputs: number;
  type: string;
  weights: Matrix;
  biases: Matrix;
  activationFunction: ActivationFunction = activationFunctions.sigmoid;
  activationFunctionDerivative: ActivationFunction =
    activationFunctionDerivatives.sigmoid;

  constructor(
    numberOfNodes: number,
    numberOfInputs: number,
    type: keyof LayerTypes,
    activationFunction: keyof activationFunctionsType = "sigmoid"
  ) {
    this.numberOfNodes = numberOfNodes;
    this.numberOfInputs = numberOfInputs;
    this.type = type;

    this.weights = new Matrix(numberOfNodes, numberOfInputs);
    this.biases = new Matrix(numberOfNodes, 1);

    this.activationFunction = activationFunctions[activationFunction];
    this.activationFunctionDerivative =
      activationFunctionDerivatives[activationFunction];
  }

  generateOutputs(inputs: Matrix) {
    let outputs = Matrix.multiply(this.weights, inputs);
    outputs.add(this.biases);
    outputs.map(this.activationFunction);

    return outputs;
  }

  generateGradient(outputs: Matrix) {
    let gradients = Matrix.map(outputs, this.activationFunctionDerivative);
    return gradients;
  }

  getWeights() {
    return this.weights;
  }

  updateWeights(adjustments: Matrix) {
    this.weights.add(adjustments);
  }
}

type LayerTypes = {
  hidden: string;
  output: string;
};
