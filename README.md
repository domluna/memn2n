# MemN2N

Implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895) with sklearn-like interface using Tensorflow. Tasks are from the [bAbl](http://arxiv.org/abs/1502.05698) dataset.

![MemN2N picture](https://www.dropbox.com/s/3rdwfxt80v45uqm/Screenshot%202015-11-19%2000.57.27.png?dl=1)

### Get Started

```
git clone git@github.com:domluna/memn2n.git

mkdir ./memn2n/data/
cd ./memn2n/data/
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar xzvf ./tasks_1-20_v1-2.tar.gz

cd ../
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

Task  |  Training Accuracy  |  Validation Accuracy  |  Testing Accuracy
------|---------------------|-----------------------|------------------
1     |  0.97               |  0.95                 |  0.96
2     |  0.84               |  0.68                 |  0.61
3     |  0.5                |  0.4                  |  0.3
4     |  0.93               |  0.94                 |  0.93
5     |  0.91               |  0.76                 |  0.81
6     |  0.78               |  0.8                  |  0.72
7     |  0.87               |  0.86                 |  0.8
8     |  0.82               |  0.8                  |  0.77
9     |  0.76               |  0.73                 |  0.72
10    |  0.71               |  0.66                 |  0.63
11    |  0.9                |  0.87                 |  0.89
12    |  0.93               |  0.92                 |  0.92
13    |  0.93               |  0.88                 |  0.93
14    |  0.88               |  0.87                 |  0.76
15    |  1.0                |  1.0                  |  1.0
16    |  0.57               |  0.42                 |  0.46
17    |  0.7                |  0.61                 |  0.57
18    |  0.93               |  0.96                 |  0.9
19    |  0.12               |  0.07                 |  0.09
20    |  1.0                |  1.0                  |  1.0

### Notes

I didn't play around with the epsilon param in Adam until after my initial results but values of 1.0 and 0.1 seem to help convergence and overfitting.
