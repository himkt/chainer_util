import json
import logging
import operator
import os
import random

logger = logging.getLogger(__file__)


def retrieve_best_epoch(args, model_dir):
    if args["epoch"] is None:
        metric = args["metric"]
        log_path = os.path.join(model_dir, "log")
        epoch, max_value = _argmax_metric(log_path, metric)
        logger.debug(f"Epoch is {epoch:04d} ({metric}: {max_value:.2f})")  # NOQA

    else:
        epoch, max_value = args["epoch"], None
        logger.debug(f"Epoch is {epoch:04d} (which is specified manually)")

    return epoch, max_value


def _argmax_metric(log_file, metric):
    op = _prepare_op(metric)
    best_epoch = 0

    if op == operator.ge:
        best_value = -1_001_001_001

    elif op == operator.le:
        best_value = 1_001_001_001

    documents = json.load(open(log_file))
    for document in documents:
        value = document[metric]
        epoch = document["epoch"]

        if op(value, best_value):
            best_epoch = epoch
            best_value = value

    return best_epoch, best_value


def _prepare_op(metric):
    ge_metrics = ["accuracy", "precision", "recall", "fscore", "f1score", "f1_score"]  # NOQA
    le_metrics = ["loss"]

    for ge_metric in ge_metrics:
        if ge_metric in metric:
            return operator.ge

    for le_metric in le_metrics:
        if le_metric in metric:
            return operator.le

    raise NotImplementedError


def set_seed(seed=31):
    import chainer
    import numpy

    logger.debug(f"Seed value: {seed}")
    if chainer.cuda.available:
        logger.debug("Fix cupy random seed")
        chainer.cuda.cupy.random.seed(seed)

    logger.debug("Fix numpy random seed")
    numpy.random.seed(seed)
    logger.debug("Fix random seed")
    random.seed(seed)
