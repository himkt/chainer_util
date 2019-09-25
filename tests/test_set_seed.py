from chainer_util import set_seed
from chainer import testing

import pytest


def get_random_vector(device, seed):
    import chainer
    import numpy
    import random
    set_seed(seed)
    x = chainer.get_device(device).xp.random.random(10)
    del chainer
    del numpy
    del random
    return x


@pytest.mark.parametrize("device", [-1, 0])
def test_set_seed(device: int):
    import chainer
    if not chainer.cuda.available and device >= 0:
        pytest.skip("GPU is not available")
    del chainer

    seed = 42
    x1 = get_random_vector(device, seed)
    x2 = get_random_vector(device, seed)
    assert sum(x1 - x2) == 0.0
