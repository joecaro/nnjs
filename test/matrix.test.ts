import Matrix from "../src/classes/Matrix/Matrix";

test("should init new matrix", () => {
  let matrix = new Matrix(1, 1);

  expect(matrix.matrix).toMatchObject([[0]]);
});

//STATIC methods
test("should create matrix from array - STATIC", () => {
  let matrix = Matrix.fromArray([0]);

  expect(matrix.matrix).toMatchObject([[0]]);
});

test("should add two matrices from - STATIC", () => {
  let a = new Matrix(1, 1);
  let b = new Matrix(1, 1);
  a.matrix[0] = [1];
  b.matrix[0] = [1];

  let matrix = Matrix.add(a, b);

  expect(matrix.matrix).toMatchObject([[2]]);
});

test("should subtract number from matrix - STATIC", () => {
  let a = new Matrix(1, 1);
  a.matrix[0] = [2];
  let b = new Matrix(1, 1);
  b.matrix[0] = [2];
  let newMatrix = Matrix.subtract(a, b);

  expect(newMatrix.matrix).toMatchObject([[0]]);
});

test("should multiply two matrices - STATIC", () => {
  let a = new Matrix(1, 1);
  a.matrix[0] = [2];
  let b = new Matrix(1, 1);
  b.matrix[0] = [2];
  let newMatrix = Matrix.multiply(a, b);

  expect(newMatrix.matrix).toMatchObject([[4]]);
});

test("should divide all matrix items by number - STATIC", () => {
  let a = new Matrix(1, 1);
  a.matrix[0] = [2];

  let newMatrix = Matrix.divNumber(a, 2);

  expect(newMatrix.matrix).toMatchObject([[1]]);
});

test("should multipy all matrix items by number - STATIC", () => {
  let a = new Matrix(1, 1);
  a.matrix[0] = [2];

  let newMatrix = Matrix.multNumber(a, 2);

  expect(newMatrix.matrix).toMatchObject([[4]]);
});

test("should perform fun on all items - STATIC", () => {
  let a = new Matrix(1, 1);
  a.matrix[0] = [2];

  let newMatrix = Matrix.map(a, (val) => val * 2);

  expect(newMatrix.matrix).toMatchObject([[4]]);
});

test("should transpose matrix - STATIC", () => {
  let a = new Matrix(2, 1);
  a.matrix[0] = [0];
  a.matrix[1] = [1];

  let newMatrix = Matrix.transpose(a);

  expect(newMatrix.matrix).toMatchObject([[0, 1]]);
});

// other methods
test("should multiply two matrices", () => {
  let a = new Matrix(1, 1);
  a.matrix[0] = [2];
  let b = new Matrix(1, 1);
  b.matrix[0] = [2];
  a.multiply(b);

  expect(a.matrix).toMatchObject([[4]]);
});

test("should add two matrices", () => {
  let a = new Matrix(1, 1);
  a.matrix[0] = [2];
  let b = new Matrix(1, 1);
  b.matrix[0] = [2];
  a.add(b);

  expect(a.matrix).toMatchObject([[4]]);
});

test("should perform math operation on all matrix items", () => {
  let a = new Matrix(1, 1);
  a.addNumber(1);
  expect(a.matrix).toMatchObject([[1]]);
  a.multNumber(2);
  expect(a.matrix).toMatchObject([[2]]);
  a.divNumber(2);
  expect(a.matrix).toMatchObject([[1]]);
});
