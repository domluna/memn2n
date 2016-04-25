# MemN2N

Implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895) with sklearn-like interface using Tensorflow. Tasks are from the [bAbl](http://arxiv.org/abs/1502.05698) dataset.

![MemN2N picture](https://www.dropbox.com/s/3rdwfxt80v45uqm/Screenshot%202015-11-19%2000.57.27.png?dl=1)

### Get Started

```
git clone git@github.com:domluna/memn2n.git
cd memn2n
python single.py
```

### Examples

Running a [single bAbI task](./single.py)

Running a [joint model on all bAbI tasks](./joint.py)

These files are also a good example of usage.

### Requirements

* tensorflow 0.8
* scikit-learn 0.17.1
* six 1.10.0

### Results

For a task to pass it has to meet 95%+ testing accuracy. Measured on single tasks on the 1k data.

pass: 1,4,12,15,20

Several other tasks have 80%+ testing accuracy.

Unless specified, the Adam optimizer was used.

The following params were used:
  * epochs: 200
  * learning_rate: 0.01
  * epsilon: 1e-8
  * embedding_size: 20

A joint model was also run with the following params:
  * epochs: 100
  * learning_rate: 0.01
  * epsilon: 1.0
  * embedding_size: 40

[Full results](./results/joint_100_epochs.csv).

### Notes

I didn't play around with the epsilon param in Adam until after my initial results but values of 1.0 and 0.1 seem to help convergence and overfitting.
