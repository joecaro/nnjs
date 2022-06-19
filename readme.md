# NNJS

nnjs is a neural network library for javascript.

### current functionality:

_Add New NN_

```js
let nn = new NN(5, 5);
```

Instantiate New Neural Network

_args_: (# of input nodes, # of output nodes, loss function)

---

_Add Hidden Layers_

```js
nn.addHiddenLayer(10, "relu");
nn.addHiddenLayer(10, "relu");
```

---

_Predict_

```js
nn.predict([1, 0]); // -> [0]
```

predict. predicts values for one array of inputs. returns array of outputs.

_args_: (inputs)

---

_Train_

```js
let inputs = [
  [1, 0],
  [0, 1],
  [1, 1],
  [0, 0],
];
let expectedValues = [[1], [1], [0], [0]];
let errors = nn.train([...inputs], [...expectedValues]);
```

train. takes in "batch" or input arrays (number[][]) and "batch" of expected values (number[][])

length of input arrays _must_ match # of input nodes
length of expected value array _must_ match # of output nodes

returns array of errors - [errorInput0, ... errorInputN]

default options = {
logBatchError: false,
logResults: true,
}

_args_: (inputs, targets, options?, batchSize = 10)

---

_Log Current State_

```js
nn.log();
//logs
// -----
// Layers:
//  Input: 2
//  Hidden Layer1:
//    Nodes: 5
//    Activation Function: sigmoid
//  Output:
//    Nodes: 1
//    Activation Function: sigmoid
// -----
// Other
//  Learning Rate: 0.1
// -----
```

Logs current state of network. Pass true to log out per layer weights.

_args_: (verbose?)

---
