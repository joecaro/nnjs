import activationFunctions, {
  activationFunctionsDerivativeType,
  activationFunctionsType,
} from "../../functions/activationFunctions";
import Node from "../../types/Node";
import Layer from "./Layer";
import { lossFunctions, lossFunctions_D } from "../../functions/lossFunctions";
import BaseLayer from "./BaseLayer";

export default class OutputLayer extends BaseLayer {
  constructor(
    numberOfNodes: number,
    previousNumberOfNodes: number = 0,
    activationFunction: keyof activationFunctionsType = "sigmoid",
    activationFunction_d: keyof activationFunctionsDerivativeType = "sigmoid_d"
  ) {
    super(
      numberOfNodes,
      previousNumberOfNodes,
      activationFunction,
      activationFunction_d
    );
  }

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
          this.activationFunction_d(weight * prevNodeActivation) *
          lossFunctions_D.mae(this.nodes[i].value, expectedValues[i]);
        let adjustment = error * gradient;
        adjustmentsArr.push(adjustment);
      });
      adjustmentsMatrix.push(adjustmentsArr);
    });

    this.proposedWeightAdjustments.push(adjustmentsMatrix);

    return this.proposedWeightAdjustments;
  }

  calculateError(compareObj: number[]) {
    if (Array.isArray(compareObj)) {
      this.nodes.forEach((node, idx) => {
        node.error = compareObj[idx] - node.value;
      });
    }
  }
}
