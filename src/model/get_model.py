from .model import *
from ..config.configuration import *



def get_specific_model(config: config):
    model = None
    if config.model_type == 'Unet':
        model = simple_unet(
            in_ch=config.in_channels,
            classes=config.out_channels)
    return model
