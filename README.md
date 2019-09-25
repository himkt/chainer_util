# Chainer Util

## Retrieve the best epoch from a log

```python
from chainer_util import retrieve_best_epoch


# `output_dir` is a directory chainer.training.Trainer creates
# Note that Trainer needs to extend chainer.training.logReport
retrieve_best_epoch({"metric": "validation/main/loss"}, output_dir)
# -> (24, 0.04) (just an example)
retrieve_best_epoch({"metric": "validation/main/fscore"}, output_dir)
# -> (32, 0.87)
```


## Fix random state

```python
from chainer_util import set_seed


set_seed(42)
```
