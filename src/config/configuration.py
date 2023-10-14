

import json

class config():

    def __init__(self):

        self.model_type = 'Unet'
        self.root_dir = r'D:\Data\Corn\b-28-split\split_folder\all-split'
        self.optimizer = 'adam'
        self.epochs = 12
        self.batch_size = 12
        self.out_channels = 3
        self.in_channels = 3

