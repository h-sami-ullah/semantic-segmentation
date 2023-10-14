from src.dataloader.data_load import *


class Datamodule(SemanticSegmentationDataLoader):
    def __init__(self, config=None, scale=1):
        self.config = config
        self.root_dir = self.config.root_dir
        self.split = None
        super().__init__(self.root_dir,self.split)

    def train_dataloader(self):
        """
        this function return a dataloader for train
        """

        trainer = SemanticSegmentationDataLoader(self.root_dir, 'train')


        return trainer

    def valid_dataloader(self):
        """
        this function return a dataloader for validation
        """
        trainer = SemanticSegmentationDataLoader(self.root_dir, 'val')


        return trainer

    def test_dataloader(self):
        """
        this function return a dataloader for test
        """
        trainer = SemanticSegmentationDataLoader(self.root_dir, 'test')


        return trainer
