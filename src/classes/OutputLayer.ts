import ActivationFunction from "../types/ActivationFunction";
import activationFunctions, {
  activationFunctionsType,
} from "../functions/activationFunctions";
import Node from "../types/Node";
import Layer from "./Layer";
import { lossFunctions, lossFunctions_D } from "../functions/lossFunctions";

export default class OutputLayer {
  nodes: Node[];
  activationFunction: ActivationFunction;
  activationFunction_d: ActivationFunction;
  weights: number[][];
  proposedWeightAdjustments: number[][][] = [];
  previousLayerNodeAmount: number;

  constructor(
    numberOfNodes: number,
    previousNumberOfNodes: number = 0,
    activationFunction: keyof activationFunctionsType = "sigmoid",
    activationFunction_d: keyof activationFunctionsType = "sigmoid_d"
  ) {
    this.nodes = new Array(numberOfNodes)
      .fill(0)
      .map(() => ({ value: 0, error: 0 }));
    this.previousLayerNodeAmount = previousNumberOfNodes;
    this.activationFunction = activationFunctions[activationFunction];
    this.activationFunction_d = activationFunctions[activationFunction_d];
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

  toArray = () => {
    return this.nodes.map((node) => node.value);
  };

  calculateError(compareObj: number[]) {
    if (Array.isArray(compareObj)) {
      this.nodes.forEach((node, idx) => {
        node.error = compareObj[idx] - node.value;
      });
    }
  }

  backProp(
    lastLayer: Layer,
    inputs: Node[],
    error: number,
    expectedValues: number[]
  ) {
    let adjustmentsMatrix: number[][] = [];

    this.weights.forEach((arr, i) => {
      let adjustmentsArr: number[] = [];
      arr.forEach((weight, j) => {
        let prevNodeActivation = lastLayer
          ? lastLayer.nodes[j].value
          : inputs[j].value;

        let gradient =
          prevNodeActivation *
          activationFunctions.sigmoid_d(weight * prevNodeActivation) *
          lossFunctions_D.mae(this.nodes[i].value, expectedValues[i]);
        let adjustment = error * gradient;
        adjustmentsArr.push(adjustment);
      });
      adjustmentsMatrix.push(adjustmentsArr);
    });

    this.proposedWeightAdjustments.push(adjustmentsMatrix);

    return this.proposedWeightAdjustments;
  }
}
