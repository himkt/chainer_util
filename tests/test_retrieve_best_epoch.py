from chainer_util import retrieve_best_epoch


def test_retrieve_best_epoch_by_loss():
    expect = (9, 1.6991147994995117)
    params = {"epoch": None, "metric": "validation/main/loss"}
    assert expect == retrieve_best_epoch(params, "tests/test_output")


def test_retrieve_best_epoch_by_fscore():
    expect = (10, 0.5683060109289617)
    params = {"epoch": None, "metric": "validation/main/fscore"}
    assert expect == retrieve_best_epoch(params, "tests/test_output")


def test_retrieve_best_epoch_by_epoch():
    expect = (1, None)
    params = {"epoch": 1}
    assert expect == retrieve_best_epoch(params, "tests/test_output")
