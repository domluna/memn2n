# MemN2N

Implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895) with sklearn-like interface using Tensorflow.

For a task to pass it has to meet 95%+ accuracy. NOTE: This is only training set accuracy, regularization is WIP so validation and testing accuracies are lower. 

Measured on single tasks on the 1k data.

pass: 1,2,5,6,7,8,9,10,11,13,14,20
fails:
- 3, ~88% acc
- 4, ~90% acc
- 12, ~90% acc
- 15, ~90% acc
- 16, ~80% acc
- 17, ~75% acc
- 18, ~75% acc
- 19, ~48% acc

## TODO

* Run a joint model
* Linear start - don't do softmax on the probabilities at first
* Adjacent sharing
* Temporal Encoding
* Work on regularization
