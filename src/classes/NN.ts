import activationFunctions, {
  activationFunctionDerivatives,
  activationFunctionStrings,
  activationFunctionsType,
} from "../functions/activationFunctions";
import lossFunctionsType, {
  lossFunctionDerivatives,
  lossFunctions,
} from "../functions/lossFunctions";
import logWithStyle from "../lib/logWithStyle";
import ActivationFunction from "../types/ActivationFunction";
import LossFunction, { LossFunctionDerivative } from "../types/LossFunction";
import Layer from "./Layer/Layer";
import Matrix from "./Matrix/Matrix";

export class NN {
  activationFunctions = activationFunctions;
  activationFunction: ActivationFunction = activationFunctions.sigmoid;
  activationFunctionDerivative: ActivationFunction =
    activationFunctionDerivatives.sigmoid;
  lossFunctions = lossFunctions;
  lossFunction: LossFunction = lossFunctions.mse;
  lossFunctionDerivative: LossFunctionDerivative = lossFunctionDerivatives.mse;

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
    this.lossFunctionDerivative = lossFunctionDerivatives[lossFunction];
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

  predict(input_array: number[], returnType: "array" | "matrix" = "array") {
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

    return returnType === "matrix"
      ? outputs
      : outputs[outputs.length - 1].toArray();
  }

  backPropogate(
    input_array: number[][],
    target_array: number[][],
    returnType: "object" | "class" = "object"
  ) {
    let weights_deltas_queue: Matrix[][] = new Array(this.layers.length)
      .fill(0)
      .map(() => []);
    let bias_deltas_queue: Matrix[][] = new Array(this.layers.length)
      .fill(0)
      .map(() => []);

    let errors_matrix = new Matrix(target_array[0].length, 1);

    input_array.forEach((arr, inputIdx) => {
      let inputs = Matrix.fromArray(arr);
      let outputs = this.predict(arr, "matrix") as Matrix[];

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

          //add errors to lists to handle back prop
          loop_errors.unshift(layer_errors);
          errors_matrix.add(layer_errors);

          //CALCULATE GRADIENTS
          let layer_gradients = layer.generateGradients(
            outputs[outputs.length - 1]
          );
          // Matrix.map(layer_errors, (v) => this.lossFunctionDerivative());
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

    errors_matrix.divNumber(input_array.length);

    return returnType === "class" ? errors_matrix : errors_matrix.toArray();
  }

  train(
    inputs: number[][],
    targets: number[][],
    options: Options = defaultOptions,
    batchSize: number = 10
  ) {
    //VALIDATE INPUTS AND TARGETS
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
      let error = this.backPropogate(
        batch_inputs,
        batch_targets,
        "class"
      ) as Matrix;

      if (options.logBatchError) error.print();
    }

    let after_prediction = this.predict(firstInputs); // get prediction after training

    if (options.logResults) {
      console.log(`Prediction before training:`);
      console.log(before_prediction);
      console.log(`Prediction after training:`);
      console.log(after_prediction);
      console.log(`Target:`);
      console.log(firstTargets);
      console.log("\nERROR AFTER TRAINING:");
      console.log(this.backPropogate([firstInputs], [firstTargets]));
    }
  }

  log(verbose: boolean = false) {
    console.log("-----\n");
    logWithStyle("Layers:", "header");
    logWithStyle(` Input:`, "subhead");
    console.log(` ${this.numberOfInputs}`);
    this.layers.forEach((layer, idx) => {
      if (layer.type === "hidden") {
        logWithStyle(` Hidden Layer${idx + 1}:`, "subhead");
        console.log(
          `   Nodes: ${layer.numberOfNodes}\n   Activation Function: ${
            activationFunctions[layer.activationFunction]
          }`
        );
      } else {
        logWithStyle(` Output:`, "subhead");
        console.log(
          `   Nodes: ${layer.numberOfNodes}\n   Activation Function: ${
            activationFunctions[layer.activationFunction]
          }`
        );
      }
    });
    console.log("\n-----\n");
    console.log(`Other`);
    console.log(` Learning Rate: ${this.learning_rate}`);
    console.log("\n-----");

    if (verbose) {
      this.layers.forEach((layer, idx) => {
        if (layer.type === "hidden") {
          console.log(` Hidden Layer${idx + 1}`);
          console.log(layer.getWeightLog());
        } else {
          console.log(` Output:`);
          console.log(layer.getWeightLog());
        }
      });
    }
  }

  toFunction() {
    let log = "";
    log += "function predict(inputs) {\n";
    log += "  activationFunctions = [\n";
    this.layers.forEach(
      (layer) =>
        (log += `   ${activationFunctionStrings[layer.activationFunction]},\n`)
    );
    log += "  ];\n\n";

    log +=
      "  function map(a, func) {\n   for (let i = 0; i < a.length; i++) {\n      for (let j = 0; j < a[0].length; j++) {\n        let val = a[i][j];\n        a[i][j] = func(val, i);\n    }\n }\n  }\n";
    log += "  let layers = [\n";
    this.layers.forEach((layer) => (log += `${layer.getWeightLog()},\n`));
    log += " ]\n\n";
    log += "  let biases = [\n";
    this.layers.map((layer) => (log += `[${layer.biases.matrix}],\n`));
    log += " ]\n";

    log += "  let outputs = [];\n\n";
    log +=
      "  layers.forEach((layer,idx) => {\n   let matrix = new Array(layer.length).fill(0).map((row) => new Array(1).fill(0).map((col) => 0)); \n\n  for (let i = 0; i < matrix.length; i++) {\n     let sum = 0;\n        layer[i].forEach((weight) => {\n        if (outputs.length>0) {\n         outputs[outputs.length - 1].forEach(\n            (input) => (sum += weight * input)\n          );\n          } else {\n            inputs.forEach((input) => (sum += weight * input));\n         }\n       });\n          matrix[i][0] = sum;\n        };";

    log += "\n\n";
    log += "    map(matrix, (v, i) => v + biases[idx][i]);\n";
    log += "\n";
    log += "    map(matrix, (v) => 1 / (1 + Math.pow(Math.E, -v)));\n";
    log += "\n";
    log += "      outputs.push(matrix);\n";
    log += "    });\n";
    log += "        return outputs[outputs.length - 1];\n";
    log += "}\n";

    console.log(log);
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

export let defaultOptions: Options = {
  logBatchError: false,
  logResults: true,
};
