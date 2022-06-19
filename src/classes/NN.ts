import activationFunctions, {
  activationFunctionDerivatives,
  activationFunctionsType,
} from "../functions/activationFunctions";
import lossFunctionsType, { lossFunctions } from "../functions/lossFunctions";
import ActivationFunction from "../types/ActivationFunction";
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

    return outputs[outputs.length - 1].toArray();
  }

  backPropogate(input_array: number[][], target_array: number[][]) {
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

    errors.divNumber(input_array.length);

    return errors;
  }

  train(
    inputs: number[][],
    targets: number[][],
    options: Options = defaultOptions,
    batchSize: number = 1
  ) {
    if (inputs.length === 0 || targets.length === 0)
      throw Error("ERROR Train - provided inputs or targets were empty");
    if (inputs.length !== targets.length)
      throw Error(
        "ERROR Train - number of input and target arrays do not match "
      );

    if (
      inputs[0].length !== this.numberOfInputs ||
      targets[0].length !== this.numberOfOutputs
    )
      throw Error(
        "ERROR Train - number of inputs and targets do not match number of inputs or outputs set for neural network"
      );

    let firstInputs = inputs[0];
    let firstTargets = targets[0];

    let before_prediction = this.predict(firstInputs); // get inital prediction

    while (inputs.length > 0) {
      let batch_inputs = inputs.splice(0, batchSize);
      let batch_targets = targets.splice(0, batchSize);
      let error = this.backPropogate(batch_inputs, batch_targets);

      if (options.logBatchError) error.print();
    }

    let after_prediction = this.predict(firstInputs); // get prediction after training

    if (options.logResults) {
      console.log(`Prediction before training: ${round(before_prediction[0])}`);
      console.log(`Prediction after training: ${round(after_prediction[0])}`);
      console.log(`Target: ${firstTargets}`);
      console.log("\nERROR AFTER TRAINING:");
      console.log(this.backPropogate([firstInputs], [firstTargets]).matrix);
    }
  }
}

function round(num: number) {
  return Math.round(num * 100) / 100;
}

function averageMatrices(matrixQueue: Matrix[]) {
  let averaged_matrix = new Matrix(matrixQueue[0].rows, matrixQueue[0].columns);

  while (matrixQueue.length > 0) {
    let workingMatrix = matrixQueue.pop() as Matrix;
    averaged_matrix.add(workingMatrix);
  }

  return averaged_matrix;
}

interface Options {
  logBatchError: boolean;
  logResults: boolean;
}

let defaultOptions: Options = {
  logBatchError: false,
  logResults: true,
};
