import activationFunctions, {
  activationFunctionDerivatives,
  activationFunctionsType,
} from "../functions/activationFunctions";
import lossFunctionsType, { lossFunctions } from "../functions/lossFunctions";
import ActivationFunction from "../types/ActivationFunction";
import LayerTypes from "../types/LayerTypes";
import LossFunction from "../types/LossFunction";
import Layer from "./Layer/Layer";
import Matrix from "./Matrix/Matrix";

export class NN {
  activationFunctions = activationFunctions;
  activationFunction: ActivationFunction = activationFunctions.sigmoid;
  activationFunctionDerivative: ActivationFunction =
    activationFunctionDerivatives.sigmoid;
  lossFunctions = lossFunctions;
  lossFunction: LossFunction = lossFunctions.mse;

  layers: Layer[] = [];

  numberOfInputs: number;
  numberOfOutputs: number;
  learning_rate: number = 0.1;

  constructor(numberOfInputs: number, numberOfOutputs: number) {
    this.numberOfInputs = numberOfInputs;
    this.numberOfOutputs = numberOfOutputs;

    let outputLayer = new Layer(
      numberOfOutputs,
      numberOfInputs,
      "output",
      "sigmoid"
    );
    this.layers.push(outputLayer);
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

  outputLayer(): Layer {
    return this.layers[this.layers.length - 1];
  }

  randomize() {
    this.layers.forEach((layer) => {
      layer.randomize();
    });
  }

  addHiddenLayer(
    numberOfNodes: number,
    activationFunction: keyof activationFunctionsType = "sigmoid"
  ) {
    let previous_ouputs =
      this.layers.length > 1
        ? this.layers[this.layers.length - 2].numberOfNodes
        : this.numberOfInputs;
    let newLayer = new Layer(
      numberOfNodes,
      previous_ouputs,
      "hidden",
      activationFunction
    );

    //insert before last layer
    this.layers.splice(-1, 0, newLayer);

    //update output layer
    this.layers[this.layers.length - 1].updateInputs(numberOfNodes);
  }

  predict(input_array: number[]) {
    let inputs = Matrix.fromArray(input_array);

    let outputs: Matrix[] = [];

    this.layers.forEach((layer, idx) => {
      if (idx === 0) {
        let layer_outputs = layer.generateOutputs(inputs);
        outputs.push(layer_outputs);
      } else {
        let layer_outputs = layer.generateOutputs(outputs[idx - 1]);
        outputs.push(layer_outputs);
      }
    });

    // // generate hidden outputs
    // let hidden = Matrix.multiply(this.weights_ih, inputs);
    // hidden.add(this.bias_h);
    // hidden.map(this.activationFunction);

    // //generate outputs
    // let outputs = Matrix.multiply(this.weights_ho, hidden);
    // outputs.add(this.bias_o);
    // outputs.map(this.activationFunction);

    return outputs[outputs.length - 1].toArray();
  }

  train(input_array: number[][], target_array: number[][]) {
    // let weights_ho_deltas_queue: Matrix[] = [];
    // let weights_ih_deltas_queue: Matrix[] = [];
    // let bias_o_deltas_queue: Matrix[] = [];
    // let bias_h_deltas_queue: Matrix[] = [];
    let weights_deltas_queue: Matrix[][] = new Array(this.layers.length)
      .fill(0)
      .map(() => []);
    let bias_deltas_queue: Matrix[][] = new Array(this.layers.length)
      .fill(0)
      .map(() => []);

    let errors = new Matrix(target_array[0].length, 1);

    input_array.forEach((arr, inputIdx) => {
      //PREDICT
      let inputs = Matrix.fromArray(arr);

      // // generate hidden outputs
      // let hidden_outputs = Matrix.multiply(this.weights_ih, inputs);
      // hidden_outputs.add(this.bias_h);
      // hidden_outputs.map(this.activationFunction);

      // //generate outputs
      // let outputs = Matrix.multiply(this.weights_ho, hidden_outputs);
      // outputs.add(this.bias_o);
      // outputs.map(this.activationFunction);

      let outputs: Matrix[] = [];

      this.layers.forEach((layer, idx) => {
        if (idx === 0) {
          let layer_outputs = layer.generateOutputs(inputs);
          outputs.push(layer_outputs);
        } else {
          let layer_outputs = layer.generateOutputs(outputs[idx - 1]);
          outputs.push(layer_outputs);
        }
      });
      //PREDICT

      let loop_errors: Matrix[] = [];
      let targets = Matrix.fromArray(target_array[inputIdx]);

      //LOOP BACKWARD THROUGH LAYERS AND GENERATE ADJUSTMENTS
      for (let i = this.layers.length - 1; i >= 0; i--) {
        let layer = this.layers[i];

        //OUTPUT LAYER
        if (layer.type === "output") {
          //CALCULATE ERRORS
          let layer_errors = Matrix.subtract(
            targets,
            outputs[outputs.length - 1]
          );
          loop_errors.unshift(errors);
          errors.add(layer_errors);

          //CALCULATE GRADIENTS
          let layer_gradients = layer.generateGradients(
            outputs[outputs.length - 1]
          );
          layer_gradients.multiply(layer_errors);
          layer_gradients.multNumber(this.learning_rate);

          //CALCULATE ADJUSTMENTS
          let previous_nodes_transposed = Matrix.transpose(
            outputs.length > 1 ? outputs[i - 1] : inputs
          );
          let weights_deltas = Matrix.multiply(
            layer_gradients,
            previous_nodes_transposed
          );

          //push delta matrices
          weights_deltas_queue[i].push(weights_deltas);
          bias_deltas_queue[i].push(layer_gradients);
          //FIRST HIDDEN LAYER
        } else if (i === 0) {
          //CALCULATE ERRORS
          let next_layer_errors = loop_errors[0];
          let next_weights_transposed = Matrix.transpose(
            this.layers[i + 1].getWeights()
          );
          let layer_errors = Matrix.multiply(
            next_weights_transposed,
            next_layer_errors
          );

          loop_errors.unshift(layer_errors);

          //CALCULATE GRADIENTS
          let layer_gradients = layer.generateGradients(outputs[i]);
          layer_gradients.multiply(layer_errors);
          layer_gradients.multNumber(this.learning_rate);

          //CALCULATE ADJUSTMENTS
          let inputs_transposed = Matrix.transpose(inputs);
          let weights_deltas = Matrix.multiply(
            layer_gradients,
            inputs_transposed
          );

          //push delta matrices
          weights_deltas_queue[i].push(weights_deltas);
          bias_deltas_queue[i].push(layer_gradients);
          //INNER HIDDEN LAYER
        } else {
          //CALCULATE ERRORS
          let next_layer_errors = loop_errors[0];
          let next_weights_transposed = Matrix.transpose(
            this.layers[i + 1].getWeights()
          );
          let layer_errors = Matrix.multiply(
            next_weights_transposed,
            next_layer_errors
          );

          loop_errors.unshift(layer_errors);

          //CALCULATE GRADIENTS
          let layer_gradients = layer.generateGradients(outputs[i]);
          layer_gradients.multiply(layer_errors);
          layer_gradients.multNumber(this.learning_rate);

          //CALCULATE ADJUSTMENTS
          let previous_nodes_transposed = Matrix.transpose(outputs[i - 1]);
          let weights_deltas = Matrix.multiply(
            layer_gradients,
            previous_nodes_transposed
          );

          //push delta matrices
          weights_deltas_queue[i].push(weights_deltas);
          bias_deltas_queue[i].push(layer_gradients);
        }
      }

      // // calculate error of outputs

      // let output_errors = Matrix.subtract(targets, outputs);

      // errors.add(output_errors);

      // // calculate gradients
      // let output_gradients = Matrix.map(
      //   outputs,
      //   this.activationFunctionDerivative
      // );
      // output_gradients.multiply(output_errors);
      // output_gradients.multNumber(this.learning_rate);

      // // Calculate weight adjustments
      // let hidden_transposed = Matrix.transpose(hidden_outputs);
      // let weights_ho_deltas = Matrix.multiply(
      //   output_gradients,
      //   hidden_transposed
      // );

      // // calculate hidden layer errors
      // let weights_ho_transposed = Matrix.transpose(this.weights_ho);
      // let hidden_erros = Matrix.multiply(weights_ho_transposed, output_errors);

      // // calculate hidden gradients
      // let hidden_gradients = Matrix.map(
      //   hidden_outputs,
      //   this.activationFunctionDerivative
      // );
      // hidden_gradients.multiply(hidden_erros);
      // hidden_gradients.multNumber(this.learning_rate);

      // // calculate hidden weight adjustments
      // let inputs_transposed = Matrix.transpose(inputs);
      // let weights_ih_deltas = Matrix.multiply(
      //   hidden_gradients,
      //   inputs_transposed
      // );

      // // add deltas to queues
      // weights_ho_deltas_queue.push(weights_ho_deltas);
      // weights_ih_deltas_queue.push(weights_ih_deltas);
      // bias_o_deltas_queue.push(output_gradients);
      // bias_h_deltas_queue.push(hidden_gradients);
    });

    // create adjustment matrices and process queues
    // average adjustments in queues then adjust appopriate matrix

    let weights_deltas: Matrix[] = [];
    let bias_deltas: Matrix[] = [];

    weights_deltas_queue.forEach((matrix_queue) => {
      let averaged_weights_deltas = averageMatrices(matrix_queue);
      weights_deltas.push(averaged_weights_deltas);
    });
    bias_deltas_queue.forEach((matrix_queue) => {
      let averaged_bias_deltas = averageMatrices(matrix_queue);
      bias_deltas.push(averaged_bias_deltas);
    });

    this.layers.forEach((layer, idx) => {
      layer.updateWeights(weights_deltas[idx]);
      layer.updateBiases(bias_deltas[idx]);
    });

    // //WHO
    // let weights_ho_deltas = new Matrix(
    //   this.weights_ho.rows,
    //   this.weights_ho.columns
    // );

    // while (weights_ho_deltas_queue.length > 0) {
    //   let workingMatrix = weights_ho_deltas_queue.pop() as Matrix;

    //   weights_ho_deltas.add(workingMatrix);
    // }
    // //WIH
    // let weights_ih_deltas = new Matrix(
    //   this.weights_ih.rows,
    //   this.weights_ih.columns
    // );

    // while (weights_ih_deltas_queue.length > 0) {
    //   let workingMatrix = weights_ih_deltas_queue.pop() as Matrix;

    //   weights_ih_deltas.add(workingMatrix);
    // }

    // //BO
    // let bias_o_deltas = new Matrix(this.bias_o.rows, this.bias_o.columns);

    // while (bias_o_deltas_queue.length > 0) {
    //   let workingMatrix = bias_o_deltas_queue.pop() as Matrix;

    //   bias_o_deltas.add(workingMatrix);
    // }

    // //BH
    // let bias_h_deltas = new Matrix(this.bias_h.rows, this.bias_h.columns);

    // while (bias_h_deltas_queue.length > 0) {
    //   let workingMatrix = bias_h_deltas_queue.pop() as Matrix;
    //   bias_h_deltas.add(workingMatrix);
    // }

    // //adjust the weights using the adjustments
    // this.weights_ho.add(weights_ho_deltas);
    // this.weights_ih.add(weights_ih_deltas);
    // // biases need to be adjusted by just the gradients
    // this.bias_o.add(bias_o_deltas);
    // this.bias_h.add(bias_h_deltas);

    // outputs.print();
    // targets.print();
    errors.divNumber(input_array.length);
    // errors.print();

    return errors;
  }
}

function averageMatrices(matrixQueue: Matrix[]) {
  let averaged_matrix = new Matrix(matrixQueue[0].rows, matrixQueue[0].columns);

  while (matrixQueue.length > 0) {
    let workingMatrix = matrixQueue.pop() as Matrix;
    averaged_matrix.add(workingMatrix);
  }

  return averaged_matrix;
}
