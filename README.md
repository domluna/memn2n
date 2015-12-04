# memn2n
WIP implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895) with sklearn-like interface using Tensorflow.

It's not 100% ready yet!

Currently doing a bunch of experiments but initial results look nice.

## Update Log

* On first challenge (1k) running for 100 epochs with a 0.01 learning rate on Adam we get low 80% accuracy.

## Things To Try

* Linear start - don't do softmax on the probabilities at first
* Adjacent sharing
* Temporal Encoding
