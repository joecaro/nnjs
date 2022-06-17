import ActivationFunction from "../types/ActivationFunction";
import activationFunctions, {
  activationFunctionsType,
} from "../functions/activationFunctions";
import Node from "../types/Node";

export default class Layer {
  nodes: Node[];
  activationFunction: ActivationFunction;
  weights: number[][];
  biases: number[];
  previousLayerNodeAmount: number;

  constructor(
    numberOfNodes: number,
    previousNumberOfNodes: number = 0,
    activationFunction: keyof activationFunctionsType
  ) {
    this.nodes = new Array(numberOfNodes).fill({ value: 0, error: 0 });
    this.previousLayerNodeAmount = previousNumberOfNodes;
    this.activationFunction = activationFunctions[activationFunction];
    this.weights = this.generateWeights();
    this.biases = new Array(numberOfNodes).fill(0);
  }

  generateWeights = () => {
    return new Array(this.nodes.length).fill(
      new Array(this.previousLayerNodeAmount).fill(0)
    );
  };

  randomizeWeights = () => {
    this.weights.forEach((arr) => {
      for (let i = 0; i < arr.length; i++) {
        arr[i] = Math.random();
      }
    });

    for (let i = 0; i < this.biases.length; i++) {
      this.biases[i] = Math.random();
    }
  };

  feedForward = (inputs: Node[]) => {
    this.nodes.forEach((node, index) => {
      // get weights assigned to this node
      let weights = this.weights[index];
      // calculate node value based on sum of prev nodes * weights
      let weightedValue = inputs.reduce(
        (a, v, idx) => v.value * weights[idx] + a,
        0
      );
      let activationValue = this.activationFunction(
        weightedValue + this.biases[index]
      );

      node.value = activationValue;
    });
  };
}
