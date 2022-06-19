import NN from "../src/index";

test("NN inits with inputs/outpus", () => {
  const nn = new NN(2, 2);
  expect(nn.numberOfInputs).toBe(2);
  expect(nn.numberOfOutputs).toBe(2);
});
test("NN adds hidden layer", () => {
  const nn = new NN(2, 2);

  expect(nn.layers.length).toBe(1);

  nn.addHiddenLayer(5);
  expect(nn.layers.length).toBe(2);

  expect(nn.outputLayer().numberOfInputs).toBe(5);
});
