export default class Matrix {
  matrix: number[][];

  constructor(rows: number, columns: number, fillValue: number = 0) {
    this.matrix = new Array(rows)
      .fill(0)
      .map((row) => new Array(columns).fill(fillValue));
  }

  static add(matrix1: number[][], matrix2: number[][]): number[][] {
    checkMatricesValidStructure(matrix1, matrix2);

    let newMatrix: number[][] = [];
    matrix1.forEach((row, i) => {
      let newRow: number[] = [];
      row.forEach((num, j) => {
        newRow.push(num + matrix2[i][j]);
      });
      newMatrix.push(newRow);
    });
    return newMatrix;
  }

  static subtract(matrix1: number[][], matrix2: number[][]): number[][] {
    checkMatricesValidStructure(matrix1, matrix2);

    let newMatrix: number[][] = [];
    matrix1.forEach((row, i) => {
      let newRow: number[] = [];
      row.forEach((num, j) => {
        newRow.push(num - matrix2[i][j]);
      });
      newMatrix.push(newRow);
    });
    return newMatrix;
  }

  static divNumber(matrix: number[][], number: number): number[][] {
    let newMatrix = matrix.map((row) => {
      return row.map((num) => num / number);
    });
    return newMatrix;
  }

  static multNumber(matrix: number[][], number: number): number[][] {
    let newMatrix = matrix.map((row) => {
      return row.map((num) => num * number);
    });
    return newMatrix;
  }

  static fromArray(array: number[][]): Matrix {
    let matrix = new Matrix(array.length, array[0].length);

    array.forEach((row, idx) => (matrix.matrix[idx] = row));

    return matrix;
  }

  add(matrix: number[][]): void {
    let newMatrix: number[][] = [];
    this.matrix.forEach((row, i) => {
      let newRow: number[] = [];
      row.forEach((num, j) => {
        newRow.push(num + matrix[i][j]);
      });
      newMatrix.push(newRow);
    });
    this.matrix = newMatrix;
  }

  divNumber(number: number): void {
    this.matrix.forEach((row) => {
      row.forEach((num) => (num /= number));
    });
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
