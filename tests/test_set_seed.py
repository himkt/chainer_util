from chainer_util import set_seed
from chainer import testing

import chainer
import pytest


def get_random_vector(device, seed):
    set_seed(seed)
    return chainer.get_device(device).xp.random.random(10)


@pytest.mark.parametrize("device", [-1, 0])
def test_set_seed(device: int):
    if not chainer.cuda.available and device >= 0:
        pytest.skip("GPU is not available")

    seed = 42
    x1 = get_random_vector(device, seed)
    x2 = get_random_vector(device, seed)
    assert testing.assert_allclose(x1, x2)
