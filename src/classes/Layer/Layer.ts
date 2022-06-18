import {
  activationFunctionsDerivativeType,
  activationFunctionsType,
} from "../../functions/activationFunctions";
import Node from "../../types/Node";
import BaseLayer from "./BaseLayer";

export default class Layer extends BaseLayer {
  biases: number[];
  gradients: number[];

  constructor(
    numberOfNodes: number,
    previousNumberOfNodes: number = 0,
    activationFunction: keyof activationFunctionsType,
    activationFunction_d: keyof activationFunctionsDerivativeType
  ) {
    super(
      numberOfNodes,
      previousNumberOfNodes,
      activationFunction,
      activationFunction_d
    );
    this.biases = new Array(numberOfNodes).fill(0).map(() => 0);
    this.gradients = new Array(numberOfNodes).fill(0).map(() => 0);
  }

  randomizeBiases = () => {
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

  backProp(prevLayerNodes: Node[], gradients: number[][]) {
    let adjustmentsMatrix: number[][] = [];

    this.weights.forEach((arr, i) => {
      let adjustmentsArr: number[] = [];
      arr.forEach((weight, j) => {
        let prevNodeActivation = prevLayerNodes[j].value;
        // sum adjustments to weights from forward layer
        let gradientSum = gradients.reduce((a, v) => a + v[j], 0);
        // console.table(gradients);

        // console.log(`gradient sum: ${gradientSum}`);

        //calculate gradient and push to tmp array
        let gradient = prevNodeActivation * gradientSum;
        adjustmentsArr.push(gradient);
      });
      adjustmentsMatrix.push(adjustmentsArr);
    });
    this.proposedWeightAdjustments.push(adjustmentsMatrix);

    return this.proposedWeightAdjustments;
  }

  calculateError(compareObj: Layer) {
    // for each node
    this.nodes.forEach((node, i) => {
      let err = 0;

      // loop through the weights to the next layer
      compareObj.weights.forEach((arr, j) => {
        // find the relative weight compared to the whole array
        let errorWeight = arr[i] / arr.reduce((a, v) => a + v, 0);

        // add the weighted error of this node
        err += errorWeight * compareObj.nodes[j].error;
      });

      node.error = err;
    });
  }
}
