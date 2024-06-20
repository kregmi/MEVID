from PIL import Image, ImageFile
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import pdb
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # tracklets | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, sample='random', seq_len=8):
        self.dataset = dataset
        self.transform = transform
        self.sample = sample
        self.seq_len = seq_len
        self.sample_methods = ['evenly', 'random']
                
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid, trackid = self.dataset[index]
        num_img = len(img_paths)
        
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = list(range(num_img))
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))
            indices = frame_indices[begin_index:end_index]
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs = list()
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                imgs.append(img)
            imgs = self.transform(imgs)
            imgs = imgs.view((self.seq_len, 3)+imgs.size()[-2:])
            return imgs, pid, camid, trackid, img_paths[0].split('/')[-1] 
            
        elif self.sample == 'evenly':
            """
            Evenly sample seq_len frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in test phase.
            """
            frame_indices = list(range(num_img))
            if len(frame_indices) > self.seq_len:
                step = len(frame_indices) // self.seq_len
                end = 0 + self.seq_len*step
                indices = frame_indices[0: end: step]
            else:
                indices = frame_indices[0:self.seq_len]
                while len(indices) < self.seq_len:
                    for index in indices:
                        if len(indices) >= self.seq_len:
                            break
                        indices.append(index)
                indices = sorted(indices)

            indices = np.array(indices)
            imgs = list()
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                imgs.append(img)
            imgs = self.transform(imgs)
            imgs = imgs.view((self.seq_len,3)+imgs.size()[-2:])
            return imgs, pid, camid, trackid, img_paths[0].split('/')[-1]       
        
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))
