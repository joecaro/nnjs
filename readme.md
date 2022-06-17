# NNJS

nnjs is a neural network library for javascript.

### current functionality:

_Add New NN_

```js
let nn = new NN(5, 5);
```

Instantiate New Neural Network

(# of input nodes, # of output nodes, loss function)

---

_Add Hidden Layers_

```js
nn.addHiddenLayer(10, "relu");
nn.addHiddenLayer(10, "relu");
```

---

_Feed Forward_

```js
nn.feedForward([1, 2, 3, 4, 5], [1, 0, 1, 1, 0]);
```

Feed forward. Calculates loss and set currnet error.

(inputs, expected outputs)

---

_Log Current State_

```js
nn.log();
//logs
// -----
// layers:
//   Input: 5
//   Hidden Layer 1: 10
//   Hidden Layer 2: 10
//   Output: 5
// -----
```
