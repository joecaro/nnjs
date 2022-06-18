import activationFunctions, {
  activationFunctionsType,
} from "../functions/activationFunctions";

import Layer from "../classes/Layer";
import OutputLayer from "./OutputLayer";
import LossFunction from "../types/LossFunction";
import lossFunctionsType, { lossFunctions } from "../functions/lossFunctions";
import { add, subtract } from "../functions/matrix";

export class NN {
  activationFunctions = activationFunctions;
  lossFunctions = lossFunctions;

  numberOfInputs: number;
  hiddenLayers: Layer[] = [];
  numberOfOutputs: number;
  outputLayer: OutputLayer;

  lossFunction: LossFunction;
  error: number = 0;
  constructor(
    numberOfInputs: number,
    numberOfOutputs: number,
    lossFunction: keyof lossFunctionsType = "mae"
  ) {
    this.numberOfInputs = numberOfInputs;
    this.numberOfOutputs = numberOfOutputs;
    this.outputLayer = new OutputLayer(numberOfOutputs, numberOfInputs);
    this.lossFunction = this.lossFunctions[lossFunction];
  }

  addHiddenLayer = (
    numberOfNodes: number,
    activationFunction: keyof activationFunctionsType = "relu"
  ) => {
    let newLayer = new Layer(
      numberOfNodes,
      this.prevNumberOfNodes(),
      activationFunction
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
      console.log(
        this.outputLayer.nodes.reduce((a, v) => a + v.error, 0) /
          this.outputLayer.nodes.length
      );
      console.log(this.error);

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
            case this.hiddenLayers.length - 1:
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

    // average adjustmentsArr
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
        weightAdjustments = add(
          this.outputLayer.proposedWeightAdjustments[i],
          weightAdjustments
        );
      }

      this.outputLayer.weights = subtract(
        this.outputLayer.weights,
        weightAdjustments
      );
    }

    // console.log(`New Weights`);
    // console.table(this.outputLayer.weights);

    this.feedForward(inputs[0], expectedValues[0], logging);

    // console.table(this.hiddenLayers[0].proposedWeightAdjustments);

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
