import NN from "../src/index";

test("NN inits with inputs/outpus", () => {
  const nn = new NN(2, 2, 2);
  expect(nn.numberOfInputs).toBe(2);
  expect(nn.numberOfOutputs).toBe(2);
});
