import activationFunctions, {
  activationFunctionsType,
} from "./activationFunctions";
import Layer from "./types/Layer";

export class NN {
  activationFunctions = activationFunctions;
  numberOfInputs: number;
  hiddenLayers: Layer[] = [];
  numberOfOutputs: number;

  outputLayer: Layer;
  error: number = 0;
  constructor(numberOfInputs: number, numberOfOutputs: number) {
    this.numberOfInputs = numberOfInputs;
    this.numberOfOutputs = numberOfOutputs;
    this.outputLayer = new Layer(numberOfOutputs, numberOfInputs, "relu");
  }

  addHiddenLayer = (
    numberOfNodes: number,
    activationFunction: keyof activationFunctionsType
  ) => {
    let newLayer = new Layer(
      numberOfNodes,
      this.prevNumberOfNodes(),
      (activationFunction = "relu")
    );
    this.hiddenLayers.push(newLayer);
  };

  prevNumberOfNodes = (): number => {
    if (this.hiddenLayers.length === 0) return this.numberOfInputs;
    else return this.hiddenLayers[this.hiddenLayers.length - 1].nodes.length;
  };

  feedForward(inputs: number[], outputs: number[]) {
    // check if we were given correct amount of inputs/outputs
    if (
      inputs.length !== this.numberOfInputs ||
      outputs.length !== this.numberOfOutputs
    ) {
      throw Error("number of inputs or outputs does not match required amount");
    }

    // formate input values
    let inputNodes = inputs.map((input) => ({ value: input, error: 0 }));

    // feed forward through hidden layers
    this.hiddenLayers.forEach((layer, idx) => {
      if (idx === 0) layer.feedForward(inputNodes);
      else layer.feedForward(this.hiddenLayers[idx - 1].nodes);
    });

    //feed forward to outputs
    this.outputLayer.feedForward(
      this.hiddenLayers[this.hiddenLayers.length - 1].nodes
    );
  }
}
