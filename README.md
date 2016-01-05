# MemN2N

Implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895) with sklearn-like interface using Tensorflow.

![MemN2N picture](https://www.dropbox.com/s/3rdwfxt80v45uqm/Screenshot%202015-11-19%2000.57.27.png?dl=1)

For a task to pass it has to meet 95%+ accuracy. NOTE: This is only training set accuracy, regularization is WIP so validation and testing accuracies are lower. 

Measured on single tasks on the 1k data.

pass: 1,2,4,5,6,7,8,9,10,11,13,14,15,17,18,20

fails:

* 3, ~88% acc
* 12, ~88% acc
* 16, ~75% acc
* 19, ~45% acc

Tasks are from the [bAbl](http://arxiv.org/abs/1502.05698) dataset.

### Usage

See `./single.py` for example usage.

### TODO

* Run a joint model
* Temporal Encoding
* Work on regularization
* Tutorial
