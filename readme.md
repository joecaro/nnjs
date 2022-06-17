# NNJS

nnjs is a neural network library for javascript.

current functionality

```
let nn = new NN(5, 5);

nn.addHiddenLayer(10, "relu");
nn.addHiddenLayer(10, "relu");

nn.feedForward([1,2,3,4,5], [1,0,1,1,0]);
```
