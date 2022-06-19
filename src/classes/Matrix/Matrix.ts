export default class Matrix {
  matrix: number[][];
  rows: number;
  columns: number;

  constructor(rows: number, columns: number, fillValue: number = 0) {
    this.matrix = new Array(rows)
      .fill(0)
      .map((row) => new Array(columns).fill(fillValue));
    this.rows = rows;
    this.columns = columns;
  }

  static add(matrix1: Matrix, matrix2: Matrix): Matrix {
    let newMatrix: Matrix = new Matrix(matrix1.rows, matrix1.columns).map(
      (val: number, i: number, j: number) =>
        matrix1.matrix[i][j] + matrix2.matrix[i][j]
    );

    return newMatrix;
  }

  static subtract(matrix1: Matrix, matrix2: Matrix): Matrix {
    let newMatrix: Matrix = new Matrix(matrix1.rows, matrix1.columns).map(
      (val: number, i: number, j: number) =>
        matrix1.matrix[i][j] - matrix2.matrix[i][j]
    );

    return newMatrix;
  }

  static multiply(a: Matrix, b: Matrix) {
    // Matrix product
    if (a.columns !== b.rows) {
      throw Error("Columns of A must match rows of B.");
    }

    return new Matrix(a.rows, b.columns).map((e, i, j) => {
      // Dot product of values in col
      let sum = 0;
      for (let k = 0; k < a.columns; k++) {
        sum += a.matrix[i][k] * b.matrix[k][j];
      }
      return sum;
    });
  }

  static divNumber(matrix: Matrix, number: number): Matrix {
    let newMatrix = new Matrix(matrix.rows, matrix.columns).map(
      (val, i, j) => matrix.matrix[i][j] / number
    );

    return newMatrix;
  }

  static multNumber(matrix: Matrix, number: number): Matrix {
    let newMatrix = new Matrix(matrix.rows, matrix.columns).map(
      (val, i, j) => matrix.matrix[i][j] * number
    );

    return newMatrix;
  }

  static fromArray(array: number[]): Matrix {
    let matrix = new Matrix(array.length, 1).map((e, i) => array[i]);

    return matrix;
  }

  static map(matrix: Matrix, func: (value: number) => number) {
    let newMatrix = new Matrix(matrix.rows, matrix.columns).map((e, i, j) =>
      func(matrix.matrix[i][j])
    );

    return newMatrix;
  }

  static transpose(matrix: Matrix) {
    return new Matrix(matrix.columns, matrix.rows).map(
      (_, i, j) => matrix.matrix[j][i]
    );
  }

  multiply(matrix: Matrix): void {
    if (this.columns !== matrix.columns || this.rows !== matrix.rows) {
      throw Error("Columns of A must match rows of B.");
    }

    this.map((e, i, j) => e * matrix.matrix[i][j]);
  }

  add(matrix: Matrix): void {
    this.map((val, i, j) => val + matrix.matrix[i][j]);
  }

  multNumber(number: number): void {
    this.matrix.forEach((row) => {
      row.forEach((num) => (num *= number));
    });
  }

  divNumber(number: number): void {
    this.matrix.forEach((row) => {
      row.forEach((num) => (num /= number));
    });
  }

  addNumber(number: number): void {
    this.matrix.forEach((row) => {
      row.forEach((num) => (num += number));
    });
  }

  toArray() {
    let arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        arr.push(this.matrix[i][j]);
      }
    }
    return arr;
  }

  map(func: (val: number, i: number, j: number) => number) {
    // Apply a function to every element of matrix
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        let val = this.matrix[i][j];
        this.matrix[i][j] = func(val, i, j);
      }
    }
    return this;
  }

  randomize() {
    return this.map((e) => Math.random() * 2 - 1);
  }

  print() {
    console.table(this.matrix);
  }
}

function checkMatricesValidStructure(matrix1: number[][], matrix2: number[][]) {
  if (matrix1.length !== matrix2.length)
    throw Error(
      "Error: Matrix.add - Cannot add matrices with different structures"
    );

  matrix1.forEach((row, idx) => {
    if (row.length !== matrix2[idx].length)
      throw Error(
        "Error: Matrix.add - Cannot add matrices with different structures"
      );
  });
}
