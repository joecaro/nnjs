export default function generateData(numberOfData: number): {
  inputs: number[][];
  expectedValues: number[][];
} {
  let inputs: number[][] = [];
  let expectedValues: number[][] = [];

  for (let i = 0; i < numberOfData; i++) {
    let x = Math.random();
    let y = Math.random();
    inputs.push([x, y]);
    expectedValues.push(x > y ? [0] : [1]);
  }

  return { inputs, expectedValues };
}
