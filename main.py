# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from src.model.get_model import *
from src.config.configuration import *
from torchsummary import summary
from src.config.configuration import *
from src.model.get_model import *
from src.optimizer.helper_optimizer import *
from src.loss.losses import *
from src.dataloader.data_load import *
from pytorch_lightning.trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from src.augmentation.image2tensor import *

def reverse_transform(inp):
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  inp = (inp * 255).astype(np.uint8)

  return inp
def main():
    configurations = config()
    model = get_specific_model(configurations)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataloader = Datamodule(configurations)
    # dataloader.config = configurations

    train_data_load = SemanticSegmentationDataLoader(configurations.root_dir,'train', True)
    #train_data_load = dataloader.train_dataloader()

    val_data_load = SemanticSegmentationDataLoader(configurations.root_dir,'val', True)


    model = model.to(device)
    print(model(torch.rand(1, 3, 512, 512)).shape)
    inputs, masks = next(iter(train_data_load))

    print(inputs)

    plt.imshow(reverse_transform(inputs[3]))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
