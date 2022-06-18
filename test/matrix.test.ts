import Matrix from "../src/classes/Matrix/Matrix";

test("should init new matrix", () => {
  let matrix = new Matrix(1, 1);

  expect(matrix.matrix).toMatchObject([[0]]);
});

test("should create matrix from static method", () => {
  let matrix = Matrix.fromArray([[0]]);

  expect(matrix.matrix).toMatchObject([[0]]);
});

test("should add two matrices from - STATIC", () => {
  let newArr = Matrix.add([[1]], [[1]]);

  expect(newArr).toMatchObject([[2]]);
});

test("should divide matrix by number - STATIC", () => {
  let newArr = Matrix.divNumber([[2]], 2);

  expect(newArr).toMatchObject([[1]]);
});

test("should multiply matrix by number - STATIC", () => {
  let newArr = Matrix.subtract([[2]], [[2]]);

  expect(newArr).toMatchObject([[4]]);
});

test("should subtract number from matrix - STATIC", () => {
  let newArr = Matrix.subtract([[2]], [[1]]);

  expect(newArr).toMatchObject([[1]]);
});
