from torchvision import datasets, transforms
from base import BaseDataLoader
from .yoga_dataset import YogaDataset, YogaDatasetTriple


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class YogaDataLoader(BaseDataLoader):
    def __init__(self, data_csv_file, batch_size, need_img=False, img_size=(220, 144), shuffle=True, validation_split=0.3, num_workers=8, training=True):
        self.dataset = YogaDataset(data_csv_file, need_img, img_size)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    

class YogaDataTripleLoader(BaseDataLoader):
    def __init__(self, data_csv_file, batch_size, need_img=False, img_size=(220, 144), shuffle=True, validation_split=0.3, num_workers=8, training=True):
        self.dataset = YogaDatasetTriple(data_csv_file, need_img, img_size)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
