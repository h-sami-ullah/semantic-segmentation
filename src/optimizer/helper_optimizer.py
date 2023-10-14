from src.config.configuration import *
from torch import optim


from typing import Union, List


def get_optimizer(config: config):
    """
    optimizer builder function

    """

    optimizer = None
    if (config.optim_type == 'adam'):
        optimizer = optim.Adam(params=params, lr=config.lr)
    elif (config.optim_type == 'sgd'):
        optimizer = optim.SGD(params=params, lr=config.lr)
    else:
        raise NotImplementedError(
            str.format('{} optimizer type is not implemented in optimizer', config.optim_type.value))
    return optimizer

