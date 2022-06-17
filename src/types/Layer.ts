import ActivationFunction from "./ActivationFunction";
import activationFunctions, {
  activationFunctionsType,
} from "../activationFunctions";
import Node from "./Node";

export default class Layer {
  nodes: Node[];
  activationFunction: ActivationFunction;
  weights: number[][];
  previousLayerNodeAmount: number;

  constructor(
    numberOfNodes: number,
    previousNumberOfNodes: number = 0,
    activationFunction: keyof activationFunctionsType
  ) {
    this.nodes = new Array(numberOfNodes).fill({ value: 0, error: 0 });
    this.previousLayerNodeAmount = previousNumberOfNodes;
    this.activationFunction = activationFunctions[activationFunction];
    this.weights = this.generateRandomWeights();
  }

  generateRandomWeights = () => {
    return new Array(this.nodes.length).fill(
      new Array(this.previousLayerNodeAmount).fill(0)
    );
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
      let activationValue = this.activationFunction(weightedValue);

      node.value = activationValue;
    });
  };
}
