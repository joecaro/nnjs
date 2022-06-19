import Layer from "../src/classes/Layer/Layer";
import Matrix from "../src/classes/Matrix/Matrix";

test("should init new layer", () => {
  let layer = new Layer(1, 1, "hidden", "sigmoid");

  expect(layer.weights.matrix).toMatchObject([[0]]);
  expect(layer.type).toBe("hidden");
});

test("should generate outputs", () => {
  let layer = new Layer(1, 1, "hidden", "sigmoid");
  layer.weights.addNumber(1);
  layer.biases.addNumber(1);

  let inputs = new Matrix(1, 1);

  let outputs = layer.generateOutputs(inputs);
  outputs.map((val) => Math.round(val * 10) / 10);

  expect(outputs.matrix).toMatchObject([[0.7]]);
});
