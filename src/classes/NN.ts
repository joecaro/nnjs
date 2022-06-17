import activationFunctions, {
  activationFunctionsType,
} from "../functions/activationFunctions";
import lossFunctions, { lossFunctionsType } from "../functions/lossFunctions";
import Layer from "../classes/Layer";
import LossFunction from "../types/LossFunction";

export class NN {
  activationFunctions = activationFunctions;
  lossFunctions = lossFunctions;

  numberOfInputs: number;
  hiddenLayers: Layer[] = [];
  numberOfOutputs: number;
  outputLayer: Layer;

  lossFunction: LossFunction;
  error: number = 0;
  constructor(
    numberOfInputs: number,
    numberOfOutputs: number,
    lossFunction: keyof lossFunctionsType = "mae"
  ) {
    this.numberOfInputs = numberOfInputs;
    this.numberOfOutputs = numberOfOutputs;
    this.outputLayer = new Layer(numberOfOutputs, numberOfInputs, "sigmoid");
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
      console.log("-----------------------");
    }
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
