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

  nn.outputLayer.weights = [[1]];
  let inputs = [1];
  nn.feedForward(inputs, [1]);

  expect(nn.outputLayer.nodes[0].value).toBe(0.7310585786300049);
  expect(nn.error).toBe(0.2689414213699951);
});

test("NN calculates node errors on feed forward", () => {
  const nn = new NN(1, 1);

  nn.outputLayer.weights = [[1]];
  let inputs = [1];
  nn.feedForward(inputs, [1]);

  expect(nn.outputLayer.nodes[0].value).toBe(0.7310585786300049);
  expect(nn.error).toBe(0.2689414213699951);
});

test("NN calculates loss", () => {
  const nn = new NN(1, 1);

  nn.calculateLoss([0], [1]);

  expect(nn.error).toBe(1);
});

test("NN logs", () => {
  const nn = new NN(5, 5);

  nn.addHiddenLayer(5, "relu");

  let message = nn.log();

  expect(message).toBe(
    `-----\nlayers:\n Inputs: 5\n Hidden Layer 1: 5\n Outputs: 5\n------`
  );
});
