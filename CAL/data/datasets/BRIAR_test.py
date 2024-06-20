# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import numpy as np
import os.path as osp
from scipy.io import loadmat
from .bases import BaseImageDataset
from collections import defaultdict
import pickle
import pdb

class BRIAR_test(BaseImageDataset):
    """
    BRIAR

    Reference:

    Dataset statistics:
    # identities: 
    # tracklets:  (train) +  (query) +  (gallery)
    # cameras: 

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).

    """

    def __init__(self, root='', verbose=True, pid_begin = 0, gallery_id = 1, **kwargs):
        super(BRIAR_test, self).__init__()
        self.dataset_dir = osp.join(root, 'briar_dawei')
        
        self.train_name_path = osp.join(self.dataset_dir, 'track_train_names.txt')
        self.query_name_path1 = osp.join(self.dataset_dir, 'track_query3.1_names.txt')
        self.query_name_path2 = osp.join(self.dataset_dir, 'track_query3.2_names.txt')

        # self.gallery_name_path = osp.join(self.dataset_dir, 'track_gallery3%d_names.txt'%gallery_id)
        self.gallery_name_path1 = osp.join(self.dataset_dir, 'track_gallery31_names.txt')
        self.gallery_name_path2 = osp.join(self.dataset_dir, 'track_gallery32_names.txt')

        self.track_train_info_path = osp.join(self.dataset_dir, 'track_train_info.txt')
        self.track_query_info_path1 = osp.join(self.dataset_dir, 'track_query3.1_info.txt')
        self.track_query_info_path2 = osp.join(self.dataset_dir, 'track_query3.2_info.txt')
        # self.track_gallery_info_path = osp.join(self.dataset_dir, 'track_gallery3%d_info.txt'%gallery_id)
        self.track_gallery_info_path1 = osp.join(self.dataset_dir, 'track_gallery31_info.txt')
        self.track_gallery_info_path2 = osp.join(self.dataset_dir, 'track_gallery32_info.txt')

        self.track_query_origin_path1 = osp.join(self.dataset_dir, 'track_query3.1_origin.txt')
        self.track_query_origin_path2 = osp.join(self.dataset_dir, 'track_query3.2_origin.txt')
        # self.track_gallery_origin_path = osp.join(self.dataset_dir, 'track_gallery3%d_origin.txt'%gallery_id)
        self.track_gallery_origin_path1 = osp.join(self.dataset_dir, 'track_gallery31_origin.txt')
        self.track_gallery_origin_path2 = osp.join(self.dataset_dir, 'track_gallery32_origin.txt')

        train_names = self.get_names(self.train_name_path)
        query_names1 = self.get_names(self.query_name_path1)
        query_names2 = self.get_names(self.query_name_path2)
        # gallery_names = self.get_names(self.gallery_name_path)
        gallery_names1 = self.get_names(self.gallery_name_path1)
        gallery_names2 = self.get_names(self.gallery_name_path2)

        query_origin1 = self.get_names(self.track_query_origin_path1)
        query_origin2 = self.get_names(self.track_query_origin_path2)
        # gallery_origin = self.get_names(self.track_gallery_origin_path)
        gallery_origin1 = self.get_names(self.track_gallery_origin_path1)
        gallery_origin2 = self.get_names(self.track_gallery_origin_path2)

        track_train = np.loadtxt(self.track_train_info_path, delimiter=',').astype('int')
        track_query1 = np.loadtxt(self.track_query_info_path1, delimiter=',').astype('int')
        track_query2 = np.loadtxt(self.track_query_info_path2, delimiter=',').astype('int')
        # track_gallery = np.loadtxt(self.track_gallery_info_path, delimiter=',').astype('int')
        track_gallery1 = np.loadtxt(self.track_gallery_info_path1, delimiter=',').astype('int')
        track_gallery2 = np.loadtxt(self.track_gallery_info_path2, delimiter=',').astype('int')

        train = self.process_data(train_names, track_train, home_dir='train')
        query1 = self.process_data(query_names1, track_query1, home_dir='query3.1')
        query2 = self.process_data(query_names2, track_query2, home_dir='query3.2')
        query = query1 + query2
        # gallery = self.process_data(gallery_names, track_gallery, home_dir='gallery3%d'%gallery_id)
        gallery1 = self.process_data(gallery_names1, track_gallery1, home_dir='gallery31')
        gallery2 = self.process_data(gallery_names2, track_gallery2, home_dir='gallery32')
        gallery = gallery1 + gallery2 

        if verbose:
            print("=> BRIAR loaded")
            self.print_dataset_statistics(train, query, gallery)
        
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

        num_clothes, pid2clothes, clothes2label = self.get_cloth_labels(train)
        self.pid2clothes, self.num_train_clothes = pid2clothes, num_clothes

    def remove_same_cloth(self, track_train_origin_path, track_train_info_path):
        new_train = []
        track_train = np.loadtxt(track_train_info_path, delimiter=',').astype('int')
        train_origin = self.get_names(track_train_origin_path)
        train_origin = np.array(train_origin)
        id_set = [row.split('/')[7] for row in train_origin]
        id_set = set(id_set)
        for subject in id_set:
            idx = [1 if row.split('/')[7] == subject else 0 for row in train_origin]
            idx = np.array(idx)
            cur_origin = train_origin[idx==1]
            cur_train = track_train[idx==1]
           
            cloth_set = [row.split('/')[-1].split('_')[1] for row in cur_origin if row.split('/')[8] == 'field']
            if 'set1' not in cloth_set:
                idx_field = [1 if row.split('/')[-1].split('_')[1] == 'set2' and row.split('/')[8] == 'field' else 0 for row in cur_origin]
                idx_control = [1 if row.split('/')[-1].split('_')[1] == 'set1' and row.split('/')[8] == 'controlled' else 0 for row in cur_origin]
            else:
                idx_field = [1 if row.split('/')[-1].split('_')[1] == 'set1' and row.split('/')[8] == 'field' else 0 for row in cur_origin]
                idx_control = [1 if row.split('/')[-1].split('_')[1] == 'set2' and row.split('/')[8] == 'controlled' else 0 for row in cur_origin]
            idx_keep = np.logical_or(np.array(idx_field), np.array(idx_control))
            cur_train = cur_train[idx_keep==1]

            if len(new_train):
                new_train = np.concatenate((new_train, cur_train), axis=0)
            else:
                new_train = cur_train
        return new_train

    def get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def process_data(self, names, meta_data, home_dir=None, relabel=True, min_seq_len=0, pid2label=None): 
        num_tracklets = meta_data.shape[0]
        #pid_list = list(set(meta_data[:, 2].tolist()))
        #if relabel:
        #    pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid, outfitid = data
            #camid = camid+(outfitid-1)*7 # relabel the camera id

            #if pid == -1:
            #    continue  # junk images are just ignored
            #assert 1 <= camid <= 6
            if pid2label:
                 pid = pid2label[pid]
            #camid -= 1  # index starts from 0
            img_names = names[start_index:end_index+1]

            # make sure image names correspond to the same person
            pnames = [img_name[1:6] for img_name in img_names]

            assert len(
                set(pnames)
            ) == 1, 'Error: a single tracklet contains different person images'

            # make sure all images are captured under the same camera
            camnames = [img_name[9] for img_name in img_names]
            assert len(
                set(camnames)
            ) == 1, 'Error: images are captured under different cameras!'

            # append image names with directory information
            img_paths = [
                osp.join(self.dataset_dir, home_dir, img_name[:6], img_name)
                for img_name in img_names
            ]

            frame_ids = [int(img_path.split('.')[-2][-5:]) for img_path in img_paths]
            track_ids = [int(img_path.split('.')[-2][-9:-6]) for img_path in img_paths]

            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid, 0))

        return tracklets

    def get_cloth_labels(self, meta_data, clothes2label = None):
        # tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        num_tracklets = len(meta_data)
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx]
            img_paths, pid, camid, outfitid = data
            # tracklet_path_list.append((img_paths, pid, camid, outfitid))
            clothes = '{}_{}'.format(pid, outfitid)
            pid_container.add(pid)
            clothes_container.add(clothes)
        
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        pid2clothes = np.zeros((num_pids, len(clothes2label)))
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx]
            img_paths, pid, camid, outfitid = data
            clothes = '{}_{}'.format(pid, outfitid)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1

        print('Number of clothes in train data: ', num_clothes)
        return num_clothes, pid2clothes, clothes2label
