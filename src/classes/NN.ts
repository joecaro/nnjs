import activationFunctions, {
  activationFunctionDerivatives,
} from "../functions/activationFunctions";
import lossFunctionsType, { lossFunctions } from "../functions/lossFunctions";
import ActivationFunction from "../types/ActivationFunction";
import LossFunction from "../types/LossFunction";
import Matrix from "./Matrix/Matrix";

export class NN {
  activationFunctions = activationFunctions;
  activationFunction: ActivationFunction = activationFunctions.sigmoid;
  activationFunctionDerivative: ActivationFunction =
    activationFunctionDerivatives.sigmoid_d;
  lossFunctions = lossFunctions;
  lossFunction: LossFunction = lossFunctions.mse;

  numberOfInputs: number;
  numberOfHiddenNodes: number;
  numberOfOutputs: number;
  learning_rate: number = 0.1;

  weights_ih: Matrix;
  weights_ho: Matrix;
  bias_h: Matrix;
  bias_o: Matrix;

  constructor(
    numberOfInputs: number,
    numberOfHiddenNodes: number,
    numberOfOutputs: number
  ) {
    this.numberOfInputs = numberOfInputs;
    this.numberOfHiddenNodes = numberOfHiddenNodes;
    this.numberOfOutputs = numberOfOutputs;

    this.weights_ih = new Matrix(numberOfHiddenNodes, numberOfInputs);
    this.weights_ho = new Matrix(numberOfOutputs, numberOfHiddenNodes);
    this.bias_h = new Matrix(numberOfHiddenNodes, 1);
    this.bias_o = new Matrix(numberOfOutputs, 1);

    this.bias_h.randomize();
    this.bias_o.randomize();
  }

  setLearning_rate(rate: number) {
    this.learning_rate = rate;
  }
  setLossFunction(lossFunction: keyof lossFunctionsType = "mse") {
    this.lossFunction = lossFunctions[lossFunction];
  }

  predict(input_array: number[]) {
    let inputs = Matrix.fromArray(input_array);

    // generate hidden outputs
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(this.activationFunction);

    //generate outputs
    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(this.activationFunction);

    return outputs.toArray();
  }

  train(input_array: number[], target_array: number[]) {
    //PREDICT
    let inputs = Matrix.fromArray(input_array);

    // generate hidden outputs
    let hidden_outputs = Matrix.multiply(this.weights_ih, inputs);
    hidden_outputs.add(this.bias_h);
    hidden_outputs.map(this.activationFunction);

    //generate outputs
    let outputs = Matrix.multiply(this.weights_ho, hidden_outputs);
    outputs.add(this.bias_o);
    outputs.map(this.activationFunction);
    //PREDICT

    // calculate error of outputs
    let targets = Matrix.fromArray(target_array);

    let output_errors = Matrix.subtract(targets, outputs);

    // calculate gradients
    let output_gradients = Matrix.map(
      outputs,
      this.activationFunctionDerivative
    );
    output_gradients.multiply(output_errors);
    output_gradients.multNumber(this.learning_rate);

    // Calculate weight adjustments
    let hidden_transposed = Matrix.transpose(hidden_outputs);
    let weight_ho_deltas = Matrix.multiply(output_gradients, hidden_transposed);

    //adjust the weights using the adjustments
    this.weights_ho.add(weight_ho_deltas);

    // biases need to be adjusted by just the gradients
    this.bias_o.add(output_gradients);

    // calculate hidden layer errors
    let weights_ho_transposed = Matrix.transpose(this.weights_ho);
    let hidden_erros = Matrix.multiply(weights_ho_transposed, output_errors);

    // calculate hidden gradients
    let hidden_gradients = Matrix.map(
      hidden_outputs,
      this.activationFunctionDerivative
    );
    hidden_gradients.multiply(hidden_erros);
    hidden_gradients.multNumber(this.learning_rate);

    // calculate hidden weight adjustments
    let inputs_transposed = Matrix.transpose(inputs);
    let weights_ih_deltas = Matrix.multiply(
      hidden_gradients,
      inputs_transposed
    );

    this.weights_ih.add(weights_ih_deltas);

    this.bias_h.add(hidden_gradients);

    // outputs.print();
    // targets.print();
    output_errors.print();

    return output_errors;
  }
}
