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

* tensorflow 1.0
* scikit-learn 0.17.1
* six 1.10.0

### Single Task Results

For a task to pass it has to meet 95%+ testing accuracy. Measured on single tasks on the 1k data.

Pass: 1,4,12,15,20

Several other tasks have 80%+ testing accuracy.

Stochastic gradient descent optimizer was used with an annealed learning rate schedule as specified in Section 4.2 of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)

The following params were used:
  * epochs: 100
  * hops: 3
  * embedding_size: 20

Task  |  Training Accuracy  |  Validation Accuracy  |  Testing Accuracy
------|---------------------|-----------------------|------------------
1     |  1.0                |  1.0                  |  1.0
2     |  1.0                |  0.86                 |  0.83
3     |  1.0                |  0.64                 |  0.54
4     |  1.0                |  0.99                 |  0.98
5     |  1.0                |  0.94                 |  0.87
6     |  1.0                |  0.97                 |  0.92
7     |  1.0                |  0.89                 |  0.84
8     |  1.0                |  0.93                 |  0.86
9     |  1.0                |  0.86                 |  0.90
10    |  1.0                |  0.80                 |  0.78
11    |  1.0                |  0.92                 |  0.84
12    |  1.0                |  1.0                  |  1.0
13    |  0.99               |  0.94                 |  0.90
14    |  1.0                |  0.97                 |  0.93
15    |  1.0                |  1.0                  |  1.0
16    |  0.81               |  0.47                 |  0.44
17    |  0.76               |  0.65                 |  0.52
18    |  0.97               |  0.96                 |  0.88
19    |  0.40               |  0.17                 |  0.13
20    |  1.0                |  1.0                  |  1.0


### Joint Training Results

Pass: 1,6,9,10,12,13,15,20

Again stochastic gradient descent optimizer was used with an annealed learning rate schedule as specified in Section 4.2 of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)

The following params were used:
  * epochs: 60
  * hops: 3
  * embedding_size: 40

Task  | Training Accuracy | Validation Accuracy |  Testing Accuracy
------|-------------------|---------------------|------------------- 
1     | 1.0               | 0.99                | 0.999
2     | 1.0               | 0.84                | 0.849
3     | 0.99              | 0.72                | 0.715
4     | 0.96              | 0.86                | 0.851
5     | 1.0               | 0.92                | 0.865
6     | 1.0               | 0.97                | 0.964
7     | 0.96              | 0.87                | 0.851
8     | 0.99              | 0.89                | 0.898
9     | 0.99              | 0.96                | 0.96
10    | 1.0               | 0.96                | 0.928
11    | 1.0               | 0.98                | 0.93
12    | 1.0               | 0.98                | 0.982
13    | 0.99              | 0.98                | 0.976
14    | 1.0               | 0.81                | 0.877
15    | 1.0               | 1.0                 | 0.983
16    | 0.64              | 0.45                | 0.44
17    | 0.77              | 0.64                | 0.547
18    | 0.85              | 0.71                | 0.586
19    | 0.24              | 0.07                | 0.104
20    | 1.0               | 1.0                 | 0.996



### Notes

Single task results are from 10 repeated trails of the single task model accross all 20 tasks with different random initializations. The performance of the model with the lowest validation accuracy for each task is shown in the table above.

Joint training results are from 10 repeated trails of the joint model accross all tasks. The performance of the single model whose validation accuracy passed the most tasks (>= 0.95) is shown in the table above (joint_scores_run2.csv). The scores from all 10 runs are located in the results/ directory.
