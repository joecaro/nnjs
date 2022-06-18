import ActivationFunction from "../types/ActivationFunction";
import activationFunctions, {
  activationFunctionsType,
} from "../functions/activationFunctions";
import Node from "../types/Node";

export default class Layer {
  nodes: Node[];
  activationFunction: ActivationFunction;
  weights: number[][];
  proposedWeightAdjustments: number[][][] = [];
  biases: number[];
  gradients: number[];
  previousLayerNodeAmount: number;

  constructor(
    numberOfNodes: number,
    previousNumberOfNodes: number = 0,
    activationFunction: keyof activationFunctionsType
  ) {
    this.nodes = new Array(numberOfNodes)
      .fill(0)
      .map(() => ({ value: 0, error: 0 }));
    this.previousLayerNodeAmount = previousNumberOfNodes;
    this.activationFunction = activationFunctions[activationFunction];
    this.weights = this.generateWeights();
    this.biases = new Array(numberOfNodes).fill(0).map(() => 0);
    this.gradients = new Array(numberOfNodes).fill(0).map(() => 0);
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

  toArray = () => {
    return this.nodes.map((node) => node.value);
  };

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
