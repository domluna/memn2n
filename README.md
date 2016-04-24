# MemN2N

Implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895) with sklearn-like interface using Tensorflow. Tasks are from the [bAbl](http://arxiv.org/abs/1502.05698) dataset.

![MemN2N picture](https://www.dropbox.com/s/3rdwfxt80v45uqm/Screenshot%202015-11-19%2000.57.27.png?dl=1)

For a task to pass it has to meet 95%+ testing accuracy. Measured on single tasks on the 1k data.

pass: 1,4,12,15,20

Several other tasks have 80%+ testing accuracy.

The next step is to run a train a model on all tasks.

### Usage

See `single.py` for example usage.

### TODO

* Run a joint model
