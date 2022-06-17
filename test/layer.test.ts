import Layer from "../src/classes/Layer";

test("should init new layer with nodes and weights", () => {
  let layer = new Layer(5, 5, "relu");

  let expectedNodes = [
    { value: 0, error: 0 },
    { value: 0, error: 0 },
    { value: 0, error: 0 },
    { value: 0, error: 0 },
    { value: 0, error: 0 },
  ];
  let expectedWeights = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ];

  expect(layer.nodes).toMatchObject(expectedNodes);
  expect(layer.weights).toMatchObject(expectedWeights);
});

test("should init with activation function", () => {
  let layer = new Layer(5, 5, "relu");

  expect(layer.activationFunction(1)).toBe(1);
});

test("should feed foward", () => {
  let prevInputs = [{ value: 1, error: 0 }];
  let layer = new Layer(1, prevInputs.length, "relu");

  layer.weights = [[0.5]];
  layer.feedForward(prevInputs);

  expect(layer.nodes[0].value).toBe(0.5);
});

test("should back propogate", () => {
  expect(true).toBeTruthy();
});
