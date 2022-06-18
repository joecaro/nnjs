import activationFunctions, {
  activationFunctionsDerivativeType,
  activationFunctionsType,
} from "../functions/activationFunctions";

import Layer from "./Layer/Layer";
import OutputLayer from "./Layer/OutputLayer";
import LossFunction from "../types/LossFunction";
import lossFunctionsType, { lossFunctions } from "../functions/lossFunctions";
import Matrix from "./Matrix/Matrix";

export class NN {
  activationFunctions = activationFunctions;
  lossFunctions = lossFunctions;

  numberOfInputs: number;
  hiddenLayers: Layer[] = [];
  numberOfOutputs: number;
  outputLayer: OutputLayer;

  lossFunction: LossFunction;
  error: number = 0;
  learningRate: number;
  constructor(
    numberOfInputs: number,
    numberOfOutputs: number,
    lossFunction: keyof lossFunctionsType = "mae",
    learningRate: number = 0.1
  ) {
    this.numberOfInputs = numberOfInputs;
    this.numberOfOutputs = numberOfOutputs;
    this.outputLayer = new OutputLayer(numberOfOutputs, numberOfInputs);
    this.lossFunction = this.lossFunctions[lossFunction];
    this.learningRate = learningRate;
  }

  addHiddenLayer = (
    numberOfNodes: number,
    activationFunction: keyof activationFunctionsType = "relu",
    activationFunction_d: keyof activationFunctionsDerivativeType = "relu_d"
  ) => {
    let newLayer = new Layer(
      numberOfNodes,
      this.prevNumberOfNodes(),
      activationFunction,
      activationFunction_d
    );
    this.hiddenLayers.push(newLayer);
    this.outputLayer.updateWeights(numberOfNodes);
  };

  prevNumberOfNodes = (): number => {
    if (this.hiddenLayers.length === 0) return this.numberOfInputs;
    else return this.hiddenLayers[this.hiddenLayers.length - 1].nodes.length;
  };

  randomizeWeights = () => {
    this.hiddenLayers.forEach((layer) => {
      layer.randomizeWeights();
    });
    this.outputLayer.randomizeWeights();
  };

  calculateLoss = (outputs: number[], expectedValues: number[]) => {
    this.error = this.lossFunction(outputs, expectedValues);
  };

  feedForward(
    inputs: number[],
    expectedValues: number[],
    logging: boolean = false
  ) {
    // check if we were given correct amount of inputs/outputs
    if (
      inputs.length !== this.numberOfInputs ||
      expectedValues.length !== this.numberOfOutputs
    ) {
      throw Error("number of inputs or outputs does not match required amount");
    }

    // format input values
    let inputNodes = inputs.map((input) => ({ value: input, error: 0 }));

    // feed forward through hidden layers
    this.hiddenLayers.forEach((layer, idx) => {
      if (idx === 0) layer.feedForward(inputNodes);
      else layer.feedForward(this.hiddenLayers[idx - 1].nodes);
    });

    //feed forward to outputs
    this.outputLayer.feedForward(
      this.hiddenLayers.length !== 0
        ? this.hiddenLayers[this.hiddenLayers.length - 1].nodes
        : inputNodes
    );

    this.calculateLoss(this.outputLayer.toArray(), expectedValues);

    this.outputLayer.calculateError(expectedValues);

    if (logging) {
      console.log("Output Layer");
      console.table(this.outputLayer.nodes);

      let guess = 0;

      this.outputLayer.nodes.forEach((node, idx) => {
        if (node.value > this.outputLayer.nodes[guess].value) {
          guess = idx;
        }
      });

      let guessedArr = new Array(this.outputLayer.nodes.length).fill(0);

      guessedArr[guess] = 1;

      console.log(
        `Expected: ${expectedValues} | guessed ${guessedArr} with ${Math.round(
          this.error * 100
        )}% loss`
      );

      console.log(`MSE: ${this.error}`);

      console.log("-----------------------");
    }
  }

  backPropogate(
    inputs: number[][],
    expectedValues: number[][],
    logging: boolean = false
  ) {
    // batch through inputs and create adjustment matrices
    inputs.forEach((inputArray, idx) => {
      this.feedForward(inputArray, expectedValues[idx], logging);

      // format input values
      let inputNodes = inputArray.map((input) => ({ value: input, error: 0 }));

      this.outputLayer.backProp(
        this.hiddenLayers[this.hiddenLayers.length - 1],
        inputNodes,
        this.error,
        expectedValues[idx]
      );

      if (this.hiddenLayers.length > 0) {
        for (let i = this.hiddenLayers.length - 1; i >= 0; i--) {
          switch (i) {
            case this.hiddenLayers.length - 1: // last hidden layer
              // array might only have one hidden layer. check length before proceeding. We need to use input nodes in only one HL
              if (this.hiddenLayers.length > 1) {
                this.hiddenLayers[i].backProp(
                  this.hiddenLayers[i - 1].nodes,
                  this.outputLayer.proposedWeightAdjustments[idx]
                );
              } else {
                this.hiddenLayers[i].backProp(
                  inputNodes,
                  this.outputLayer.proposedWeightAdjustments[idx]
                );
              }
            case 0:
              if (this.hiddenLayers.length > 1) {
                this.hiddenLayers[i].backProp(
                  inputNodes,
                  this.hiddenLayers[i + 1].proposedWeightAdjustments[idx]
                );
              } else {
                this.hiddenLayers[i].backProp(
                  inputNodes,
                  this.outputLayer.proposedWeightAdjustments[idx]
                );
              }
            case -1:
              break;
            default:
              this.hiddenLayers[i].backProp(
                this.hiddenLayers[i - 1].nodes,
                this.hiddenLayers[i + 1].proposedWeightAdjustments[idx]
              );
          }
        }
      }
    });

    // console.log(`Current Weights`);
    // console.table(this.outputLayer.weights);
    // console.log(`\nProposed Weight Adjustments`);
    // console.table(this.outputLayer.proposedWeightAdjustments);

    // average output adjustmentsArr
    if (this.outputLayer.proposedWeightAdjustments.length > 0) {
      let weightAdjustments: number[][] = new Array(
        this.outputLayer.proposedWeightAdjustments[0].length
      )
        .fill(0)
        .map((row) =>
          new Array(this.outputLayer.proposedWeightAdjustments[0][0].length)
            .fill(0)
            .map(() => 0)
        );

      for (
        let i = 0;
        i < this.outputLayer.proposedWeightAdjustments.length;
        i++
      ) {
        weightAdjustments = Matrix.add(
          this.outputLayer.proposedWeightAdjustments[i],
          weightAdjustments
        );
      }

      weightAdjustments = Matrix.multNumber(
        weightAdjustments,
        this.learningRate
      );

      this.outputLayer.weights = Matrix.subtract(
        this.outputLayer.weights,
        weightAdjustments
      );

      this.outputLayer.proposedWeightAdjustments = [];
    }

    // average adjustments for each hidden layer

    if (this.hiddenLayers.length > 0) {
      for (let i = 0; i < this.hiddenLayers.length; i++) {
        let hiddenLayer = this.hiddenLayers[i];
        let weightAdjustments = new Matrix(
          hiddenLayer.weights.length,
          hiddenLayer.weights[0].length
        ).matrix;

        for (let i = 0; i < hiddenLayer.proposedWeightAdjustments.length; i++) {
          weightAdjustments = Matrix.add(
            hiddenLayer.proposedWeightAdjustments[i],
            weightAdjustments
          );
        }

        weightAdjustments = Matrix.multNumber(
          weightAdjustments,
          this.learningRate
        );

        hiddenLayer.weights = Matrix.subtract(
          hiddenLayer.weights,
          weightAdjustments
        );

        hiddenLayer.proposedWeightAdjustments = [];
      }
    }
    this.feedForward(inputs[0], expectedValues[0], logging);

    console.log(`Values After Quick Train`);
    console.table(this.outputLayer.nodes);
  }

  log = () => {
    let message = `-----\nlayers:\n Inputs: ${this.numberOfInputs}`;

    this.hiddenLayers.forEach((layer, idx) => {
      message += `\n Hidden Layer ${idx + 1}: ${layer.nodes.length}`;
    });

    message += `\n Outputs: ${this.numberOfOutputs}\n------`;

    console.log(message);
    return message;
  };
}
