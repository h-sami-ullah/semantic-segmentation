from src.dataloader.helper import *
from src.config.configuration import *
from src.model.get_model import *
from src.optimizer.helper_optimizer import *
from src.loss.losses import *
import torch.nn as nn
from pytorch_lightning.trainer import Trainer

def train_model(configurations):
    dataloader = Datamodule(configurations.root_dir)
    train_data_load = dataloader.train_dataloader()

    val_data_load = dataloader.valid_dataloader()

    test_data_load = dataloader.test_dataloader()
    optimizer = get_optimizer(configurations)
    criterion = nn.CrossEntropyLoss() if config.out_channels > 1 else nn.BCEWithLogitsLoss()
    model  = get_specific_model(configurations)
    print(model)
    trainer = Trainer(optimizer, criterion)

    trainer.fit(model, train_data_load, val_data_load)

if __name__=='__main__':
    configurations = config()
    train_model(configurations)









