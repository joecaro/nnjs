import ActivationFunction from "../../types/ActivationFunction";
import activationFunctions, {
  activationFunctionDerivs,
  activationFunctionsDerivativeType,
  activationFunctionsType,
} from "../../functions/activationFunctions";
import Node from "../../types/Node";

export default abstract class BaseLayer {
  nodes: Node[];
  activationFunction: ActivationFunction;
  activationFunction_d: ActivationFunction;
  weights: number[][];
  proposedWeightAdjustments: number[][][] = [];
  previousLayerNodeAmount: number;

  constructor(
    numberOfNodes: number,
    previousNumberOfNodes: number = 0,
    activationFunction: keyof activationFunctionsType,
    activationFunction_d: keyof activationFunctionsDerivativeType
  ) {
    this.nodes = new Array(numberOfNodes)
      .fill(0)
      .map(() => ({ value: 0, error: 0 }));
    this.previousLayerNodeAmount = previousNumberOfNodes;
    this.activationFunction = activationFunctions[activationFunction];
    this.activationFunction_d = activationFunctionDerivs[activationFunction_d];
    this.weights = this.generateWeights();
  }

  generateWeights = () => {
    return new Array(this.nodes.length)
      .fill(0)
      .map(() => new Array(this.previousLayerNodeAmount).fill(0).map(() => 0));
  };

  updateWeights = (numberOfPrevNodes: number) => {
    this.weights = new Array(this.nodes.length)
      .fill(0)
      .map(() => new Array(numberOfPrevNodes).fill(0).map(() => 0));
  };

  randomizeWeights = () => {
    for (let i = 0; i < this.weights.length; i++) {
      for (let j = 0; j < this.weights[i].length; j++) {
        this.weights[i][j] = Math.random();
      }
    }
  };

  toArray = () => {
    return this.nodes.map((node) => node.value);
  };
}
