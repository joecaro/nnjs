class Matrix {
  matrix: number[][];

  constructor(rows: number, columns: number, fillValue: number = 0) {
    this.matrix = new Array(rows)
      .fill(0)
      .map((row) => new Array(columns).fill(fillValue));
  }

  add(matrix1: number[][], matrix2: number[][]): number[][] {
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

  div(matrix: number[][], number: number) {
    matrix.forEach((row) => {
      row.forEach((num) => (num /= number));
    });
  }
}
