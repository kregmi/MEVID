# encoding: utf-8
"""
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
from PIL import Image
import math

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

class BRIAR(BaseImageDataset):
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

    def __init__(self, root='', verbose=True, pid_begin = 0, gallery_id = 1, seq_len=16, stride=4, **kwargs):
        super(BRIAR, self).__init__()
        self.dataset_dir = osp.join(root, 'briar_dawei') # dataset path
        self.train_name_path = osp.join(self.dataset_dir, 'track_train_names.txt')
        self.train_name_path2 = osp.join(self.dataset_dir, 'track_add_names.txt')
        self.train_name_path3 = osp.join(self.dataset_dir, 'track_val_names.txt')
        self.train_name_path4 = osp.join(self.dataset_dir, 'track_BRS3_names.txt')
        self.train_name_path5 = osp.join(self.dataset_dir, 'track_BRS32_names.txt')
        self.train_name_path6 = osp.join(self.dataset_dir, 'track_BRC_new_names.txt')
        self.train_name_path7 = osp.join(self.dataset_dir, 'track_BRS4_names.txt')
        self.track_train_info_path = osp.join(self.dataset_dir, 'track_train_info.txt')
        self.track_train_info_path2 = osp.join(self.dataset_dir, 'track_add_info.txt')
        self.track_train_info_path3 = osp.join(self.dataset_dir, 'track_val_info.txt')
        self.track_train_info_path4 = osp.join(self.dataset_dir, 'track_BRS3_info.txt')
        self.track_train_info_path5 = osp.join(self.dataset_dir, 'track_BRS32_info.txt')
        self.track_train_info_path6 = osp.join(self.dataset_dir, 'track_BRC_new_info.txt')
        self.track_train_info_path7 = osp.join(self.dataset_dir, 'track_BRS4_info.txt')
        
        self.track_train_origin_path = osp.join(self.dataset_dir, 'track_train_origin.txt')
        self.track_train_origin_path2 = osp.join(self.dataset_dir, 'track_add_origin.txt')
        self.track_train_origin_path3 = osp.join(self.dataset_dir, 'track_val_origin.txt')
        self.track_train_origin_path4 = osp.join(self.dataset_dir, 'track_BRS3_origin.txt')
        self.track_train_origin_path5 = osp.join(self.dataset_dir, 'track_BRS32_origin.txt')
        self.track_train_origin_path7 = osp.join(self.dataset_dir, 'track_BRS4_origin.txt')

        self.query_name_path1 = osp.join(self.dataset_dir, 'track_query3.1_names.txt')
        self.query_name_path2 = osp.join(self.dataset_dir, 'track_query3.2_names.txt')
        self.gallery_name_path1 = osp.join(self.dataset_dir, 'track_gallery31_names.txt')
        self.gallery_name_path2 = osp.join(self.dataset_dir, 'track_gallery32_names.txt')
        self.track_query_info_path1 = osp.join(self.dataset_dir, 'track_query3.1_info.txt')
        self.track_query_info_path2 = osp.join(self.dataset_dir, 'track_query3.2_info.txt')
        self.track_gallery_info_path1 = osp.join(self.dataset_dir, 'track_gallery31_info.txt')
        self.track_gallery_info_path2 = osp.join(self.dataset_dir, 'track_gallery32_info.txt')
        self.track_query_origin_path1 = osp.join(self.dataset_dir, 'track_query3.1_origin.txt')
        self.track_query_origin_path2 = osp.join(self.dataset_dir, 'track_query3.2_origin.txt')
        self.track_gallery_origin_path1 = osp.join(self.dataset_dir, 'track_gallery31_origin.txt')
        self.track_gallery_origin_path2 = osp.join(self.dataset_dir, 'track_gallery32_origin.txt')

        train_names = self.get_names(self.train_name_path)
        train_names2 = self.get_names(self.train_name_path2)
        train_names3 = self.get_names(self.train_name_path3)
        train_names4 = self.get_names(self.train_name_path4)
        train_names5 = self.get_names(self.train_name_path5)
        train_names6 = self.get_names(self.train_name_path6)
        train_names7 = self.get_names(self.train_name_path7)
        
        train_origin = self.get_names(self.track_train_origin_path)
        train_origin2 = self.get_names(self.track_train_origin_path2)
        train_origin3 = self.get_names(self.track_train_origin_path3)

        track_train = np.loadtxt(self.track_train_info_path, delimiter=',').astype('int')
        #train_origin_idx = [1 if row.split('/')[-6] != 'BRS2' else 0 for row in train_origin]
        #train_origin_idx = np.array(train_origin_idx)
        #track_train = track_train[train_origin_idx==1]
        track_train2 = np.loadtxt(self.track_train_info_path2, delimiter=',').astype('int')
        #train_origin_idx = [1 if row.split('/')[-6] != 'BRS2' else 0 for row in train_origin2]
        #train_origin_idx = np.array(train_origin_idx)
        #track_train2 = track_train2[train_origin_idx==1]
        track_train3 = np.loadtxt(self.track_train_info_path3, delimiter=',').astype('int')
        #train_origin_idx = [1 if row.split('/')[-6] != 'BRS2' else 0 for row in train_origin3]
        #train_origin_idx = np.array(train_origin_idx)
        #track_train3 = track_train3[train_origin_idx==1]
        
        # remove field and controlled data with the same outfit
        track_train4 = self.remove_same_cloth(self.track_train_origin_path4, self.track_train_info_path4)
        track_train5 = self.remove_same_cloth(self.track_train_origin_path5, self.track_train_info_path5)
        track_train6 = np.loadtxt(self.track_train_info_path6, delimiter=',').astype('int')
        track_train7 = self.remove_same_cloth(self.track_train_origin_path7, self.track_train_info_path7)
        
        # relabel pids
        meta_data = np.concatenate((track_train, track_train2, track_train3, track_train4, track_train5, track_train6, track_train7), axis=0) # select the subsets we use
        pid_list = list(set(meta_data[:, 2].tolist()))
        pid2label = {pid: label for label, pid in enumerate(pid_list)}
        
        train1 = self.process_data(train_names, track_train, home_dir='train', pid2label=pid2label)
        train2 = self.process_data(train_names2, track_train2, home_dir='add', pid2label=pid2label)
        train3 = self.process_data(train_names3, track_train3, home_dir='val', pid2label=pid2label)
        train4 = self.process_data(train_names4, track_train4, home_dir='BRS3', pid2label=pid2label)
        train5 = self.process_data(train_names5, track_train5, home_dir='BRS32', pid2label=pid2label)
        train6 = self.process_data(train_names6, track_train6, home_dir='BRC_new', pid2label=pid2label)
        train7 = self.process_data(train_names7, track_train7, home_dir='BRS4', pid2label=pid2label)
        train = train1 + train2 + train3 + train4 + train5 + train6 + train7# load all the training subsets, uncoment the above unused variables

        query_names1 = self.get_names(self.query_name_path1)
        query_names2 = self.get_names(self.query_name_path2)
        gallery_names1 = self.get_names(self.gallery_name_path1)
        gallery_names2 = self.get_names(self.gallery_name_path2)

        gallery_origin1 = self.get_names(self.track_gallery_origin_path1)
        gallery_origin2 = self.get_names(self.track_gallery_origin_path2)
        gallery_idx1 = [1 if row.split('/')[-2] == 'wb' else 0 for row in gallery_origin1]
        gallery_idx1 = np.array(gallery_idx1)
        gallery_idx2 = [1 if row.split('/')[-2] == 'wb' else 0 for row in gallery_origin2]
        gallery_idx2 = np.array(gallery_idx2)

        track_query1 = np.loadtxt(self.track_query_info_path1, delimiter=',').astype('int')
        track_query2 = np.loadtxt(self.track_query_info_path2, delimiter=',').astype('int')
        track_gallery1 = np.loadtxt(self.track_gallery_info_path1, delimiter=',').astype('int')
        track_gallery2 = np.loadtxt(self.track_gallery_info_path2, delimiter=',').astype('int')
        # remove face modality
        track_gallery1 = track_gallery1[gallery_idx1==1]
        track_gallery2 = track_gallery2[gallery_idx2==1]

        query1 = self.process_data(query_names1, track_query1, home_dir='query3.1')
        query2 = self.process_data(query_names2, track_query2, home_dir='query3.2')
        query = query1 + query2
        gallery1 = self.process_data(gallery_names1, track_gallery1, home_dir='gallery31')
        gallery2 = self.process_data(gallery_names2, track_gallery2, home_dir='gallery32')
        gallery = gallery1 + gallery2
        # # 10% data for validation
        query = query[::10]
        gallery = gallery[::10]

        recombined_query, query_vid2clip_index = self._recombination_for_testset(query, seq_len=seq_len, stride=stride)
        recombined_gallery, gallery_vid2clip_index = self._recombination_for_testset(gallery, seq_len=seq_len, stride=stride)
        self.recombined_query = recombined_query
        self.recombined_gallery = recombined_gallery
        self.query_vid2clip_index = query_vid2clip_index
        self.gallery_vid2clip_index = gallery_vid2clip_index


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
            #track_ids = [int(img_path.split('.')[-2][-9:-6]) for img_path in img_paths]
                
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid, outfitid))
                
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

    def _recombination_for_testset(self, dataset, seq_len=16, stride=4):
        ''' Split all videos in test set into lots of equilong clips.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride

        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, (img_paths, pid, camid, clothes_id) in enumerate(dataset):
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            for i in range(len(img_paths)//(seq_len*stride)):
                for j in range(stride):
                    begin_idx = i * (seq_len * stride) + j
                    end_idx = (i + 1) * (seq_len * stride)
                    clip_paths = img_paths[begin_idx : end_idx : stride]
                    assert(len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))

            # process the remaining sequence that can't be divisible by seq_len*stride        
            if len(img_paths)%(seq_len*stride) != 0:
                # reducing stride
                new_stride = (len(img_paths)%(seq_len*stride)) // seq_len
                for i in range(new_stride):
                    begin_idx = len(img_paths) // (seq_len*stride) * (seq_len*stride) + i
                    end_idx = len(img_paths) // (seq_len*stride) * (seq_len*stride) + seq_len * new_stride
                    clip_paths = img_paths[begin_idx : end_idx : new_stride]
                    assert(len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
                # # process the remaining sequence that can't be divisible by seq_len
                if len(img_paths) % seq_len != 0:
                    clip_paths = img_paths[len(img_paths)//seq_len*seq_len:]
                    # loop padding
                    while len(clip_paths) < seq_len:
                        for index in clip_paths:
                            if len(clip_paths) >= seq_len:
                                break
                            # clip_paths.append(index)
                            clip_paths = clip_paths + (index,)
                    assert(len(clip_paths) == seq_len)
                    # new_dataset.append((clip_paths, pid, camid, clothes_id))
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert((vid2clip_index[idx, 1]-vid2clip_index[idx, 0]) == math.floor(len(img_paths)/seq_len))

        return new_dataset, vid2clip_index.tolist()

