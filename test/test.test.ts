import NN from "../src/index";

test("NN inits with inputs/outpus", () => {
  const nn = new NN(5, 5);
  expect(nn.numberOfInputs).toBe(5);
  expect(nn.numberOfOutputs).toBe(5);
});
test("NN adds first hidden layer", () => {
  const nn = new NN(5, 5);

  nn.addHiddenLayer(1, "relu");

  expect(nn.hiddenLayers.length).toBe(1);
});
test("NN adds multiple hidden layers", () => {
  const nn = new NN(5, 5);

  nn.addHiddenLayer(3, "relu");
  nn.addHiddenLayer(1, "relu");

  expect(nn.hiddenLayers.length).toBe(2);
});
test("NN feeds forward", () => {
  const nn = new NN(1, 1);

  nn.addHiddenLayer(1, "relu");
  nn.addHiddenLayer(1, "relu");

  nn.hiddenLayers[0].weights = [[1]];
  nn.hiddenLayers[1].weights = [[0.5]];
  nn.outputLayer.weights = [[1]];
  let inputs = [1];
  nn.feedForward(inputs, [1]);

  expect(nn.outputLayer.nodes[0].value).toBe(0.5);
});
test("NN logs", () => {
  const nn = new NN(5, 5);

  nn.addHiddenLayer(5, "relu");

  let message = nn.log();

  expect(message).toBe(
    `-----\nlayers:\n Inputs: 5\n Hidden Layer 1: 5\n Outputs: 5\n------`
  );
});
