# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .veri import VeRi
from .dataset_loader import ImageDataset
from .bases import BaseImageDataset

__factory = {
    'market1501': Market1501,
    # 'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'veri': VeRi,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)

def init_datasets(names, *args, **kwargs):
    datasets = []
    for name in names:
        datasets.append(init_dataset(name, *args, **kwargs))

    return datasets

class Multidataset(BaseImageDataset):
    def __init__(self, datasets):
        self.train = []
        self.query = []
        self.gallery = []
        for dataset in datasets:
            self.train += dataset.train
            self.query += dataset.query
            self.gallery += dataset.gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

def init_multidataset(datasets):
    return Multidataset(datasets)