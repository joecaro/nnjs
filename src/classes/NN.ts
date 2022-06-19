import activationFunctions, {
  activationFunctionDerivatives,
  activationFunctionsType,
} from "../functions/activationFunctions";
import lossFunctionsType, { lossFunctions } from "../functions/lossFunctions";
import ActivationFunction from "../types/ActivationFunction";
import LossFunction from "../types/LossFunction";
import Matrix from "./Matrix/Matrix";

export class NN {
  activationFunctions = activationFunctions;
  activationFunction: ActivationFunction = activationFunctions.sigmoid;
  activationFunctionDerivative: ActivationFunction =
    activationFunctionDerivatives.sigmoid;
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
  setActivationFunction(activationFunction: keyof activationFunctionsType) {
    this.activationFunction = activationFunctions[activationFunction];
    this.activationFunctionDerivative =
      activationFunctionDerivatives[activationFunction];
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

  train(input_array: number[][], target_array: number[][]) {
    let weights_ho_deltas_queue: Matrix[] = [];
    let weights_ih_deltas_queue: Matrix[] = [];
    let bias_o_deltas_queue: Matrix[] = [];
    let bias_h_deltas_queue: Matrix[] = [];
    let errors = new Matrix(target_array[0].length, 1);

    input_array.forEach((arr, inputIdx) => {
      //PREDICT
      let inputs = Matrix.fromArray(arr);

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
      let targets = Matrix.fromArray(target_array[inputIdx]);

      let output_errors = Matrix.subtract(targets, outputs);

      errors.add(output_errors);

      // calculate gradients
      let output_gradients = Matrix.map(
        outputs,
        this.activationFunctionDerivative
      );
      output_gradients.multiply(output_errors);
      output_gradients.multNumber(this.learning_rate);

      // Calculate weight adjustments
      let hidden_transposed = Matrix.transpose(hidden_outputs);
      let weights_ho_deltas = Matrix.multiply(
        output_gradients,
        hidden_transposed
      );

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

      // add deltas to queues
      weights_ho_deltas_queue.push(weights_ho_deltas);
      weights_ih_deltas_queue.push(weights_ih_deltas);
      bias_o_deltas_queue.push(output_gradients);
      bias_h_deltas_queue.push(hidden_gradients);
    });

    // create adjustment matrices and process queues
    // average adjustments in queues then adjust appopriate matrix

    //WHO
    let weights_ho_deltas = new Matrix(
      this.weights_ho.rows,
      this.weights_ho.columns
    );

    while (weights_ho_deltas_queue.length > 0) {
      let workingMatrix = weights_ho_deltas_queue.pop() as Matrix;

      weights_ho_deltas.add(workingMatrix);
    }
    //WIH
    let weights_ih_deltas = new Matrix(
      this.weights_ih.rows,
      this.weights_ih.columns
    );

    while (weights_ih_deltas_queue.length > 0) {
      let workingMatrix = weights_ih_deltas_queue.pop() as Matrix;

      weights_ih_deltas.add(workingMatrix);
    }

    //BO
    let bias_o_deltas = new Matrix(this.bias_o.rows, this.bias_o.columns);

    while (bias_o_deltas_queue.length > 0) {
      let workingMatrix = bias_o_deltas_queue.pop() as Matrix;

      bias_o_deltas.add(workingMatrix);
    }

    //BH
    let bias_h_deltas = new Matrix(this.bias_h.rows, this.bias_h.columns);

    while (bias_h_deltas_queue.length > 0) {
      let workingMatrix = bias_h_deltas_queue.pop() as Matrix;
      bias_h_deltas.add(workingMatrix);
    }

    //adjust the weights using the adjustments
    this.weights_ho.add(weights_ho_deltas);
    this.weights_ih.add(weights_ih_deltas);
    // biases need to be adjusted by just the gradients
    this.bias_o.add(bias_o_deltas);
    this.bias_h.add(bias_h_deltas);

    // outputs.print();
    // targets.print();
    errors.divNumber(input_array.length);
    // errors.print();

    return errors;
  }
}
