import os
import json
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


class RCCDataset(Dataset):

    def __init__(self, cfg, split):
        self.cfg = cfg

        print('Speaker Dataset loading vocab json file: ', cfg.data.vocab_json)
        self.vocab_json = cfg.data.vocab_json
        self.word_to_idx = json.load(open(self.vocab_json, 'r'))
        self.idx_to_word = {}
        for word, idx in self.word_to_idx.items():
            self.idx_to_word[idx] = word
            
        self.vocab_size = len(self.idx_to_word)
        print('vocab size is ', self.vocab_size)

        self.split = split
        self.IGNORE = -1
    
        data_path = 'data/clevr-change'

        self.d_img_path = '{}/features/features'.format(data_path)
        self.sc_img_path = '{}/features/sc_features'.format(data_path)
        self.nsc_img_path = '{}/features/nsc_features'.format(data_path)

        if split == 'train': # 1-to-1
            self.batch_size = cfg.data.train.batch_size
            self.captions = np.load('{}/train_caps.npy'.format(data_path))
            self.ids = json.load(open('{}/train_ids.json'.format(data_path), 'r'))

            self.nsc_caption = np.load(os.path.join(data_path, 'train_nsc_caps.npy'))
            """
            {'there is no change':0, 
            'nothing was modified':1, 
            'no change was made':2, 
            'the scene remains the same':3,
            'nothing has changed':4,
            'there is no difference':5, 
            'the scene is the same as before':6, 
            'no change has occurred':7, 
            'the two scenes seem identical':8}
            """
            self.nsc_label = np.concatenate((self.nsc_caption[:, 1:], np.zeros((len(self.nsc_caption), 1), dtype=np.int64)), axis=1)
            m = self.nsc_label == 0
            self.nsc_label[m] = self.IGNORE
            self.nsc_mask = (self.nsc_caption > 0).astype(np.int64)

            self.train_nsc_caps_idx = json.load(open('{}/train_nsc_caps_idx.json'.format(data_path)))


        elif split == 'val':
            self.batch_size = cfg.data.val.batch_size
            self.ids = json.load(open('{}/val_ids.json'.format(data_path),'r'))

        elif split == 'test':
            self.batch_size = cfg.data.test.batch_size
            self.ids = json.load(open('{}/test_ids.json'.format(data_path),'r'))            

    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index): 
        img_id = self.ids[index]
        img_id_ = img_id.replace('png', 'npy')
        
        if random.random() < 0.5 and self.split == 'train':
            d_img = torch.FloatTensor(np.load(os.path.join(self.nsc_img_path, img_id_.replace('default', 'nonsemantic')))) 
            nsc_img = torch.FloatTensor(np.load(os.path.join(self.d_img_path, img_id_)))                       
        else:
            d_img = torch.FloatTensor(np.load(os.path.join(self.d_img_path, img_id_)))
            nsc_img = torch.FloatTensor(np.load(os.path.join(self.nsc_img_path, img_id_.replace('default', 'nonsemantic'))))   
        
        sc_img = torch.FloatTensor(np.load(os.path.join(self.sc_img_path, img_id_.replace('default', 'semantic'))))

        if self.split == 'train':
            sc_caption = self.captions[index]
            sc_mask = (sc_caption > 0).astype(np.int32)
            sc_label = [self.IGNORE if m == 0 else tid for tid, m in zip(sc_caption.tolist(), sc_mask.tolist())][1:] + [self.IGNORE]
            sc_label = np.array(sc_label)

            i = random.choice(self.train_nsc_caps_idx[img_id]) 

            nsc_caption = self.nsc_caption[i]
            nsc_label = self.nsc_label[i]
            nsc_mask = self.nsc_mask[i]

            return (d_img, sc_img, nsc_img, sc_caption, sc_label, sc_mask, nsc_caption, nsc_label, nsc_mask)
        else:
            return (d_img, sc_img, nsc_img, img_id)


    def get_vocab_size(self):
        return self.vocab_size

    def get_idx_to_word(self):
        return self.idx_to_word

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_max_seq_length(self):
        return self.max_seq_length


def rcc_collate(batch):
    transposed = list(zip(*batch))

    d_img = transposed[0]
    sc_img = transposed[1]
    nsc_img = transposed[2]
    
    d_img = default_collate(d_img)
    sc_img = default_collate(sc_img)
    nsc_img = default_collate(nsc_img)     

    sc_caption_batch = default_collate(transposed[3])
    sc_label_batch = default_collate(transposed[4])
    sc_mask_batch = default_collate(transposed[5])

    nsc_caption_batch = default_collate(transposed[6])
    nsc_label_batch = default_collate(transposed[7])
    nsc_mask_batch = default_collate(transposed[8])

    max_length = sc_mask_batch.sum(-1).max(-1).values.data
    sc_caption_batch = sc_caption_batch[:, :max_length]
    sc_label_batch = sc_label_batch[:, :max_length]
    sc_mask_batch = sc_mask_batch[:, :max_length]

    return (d_img, sc_img, nsc_img, sc_caption_batch, sc_label_batch, sc_mask_batch, nsc_caption_batch, nsc_label_batch, nsc_mask_batch)


def rcc_collate_test(batch):
    transposed = list(zip(*batch))

    d_img = transposed[0]
    sc_img = transposed[1]
    nsc_img = transposed[2]

    d_img = default_collate(d_img)
    sc_img = default_collate(sc_img)
    nsc_img = default_collate(nsc_img)  

    id_batch = transposed[3]

    return (d_img, sc_img, nsc_img, id_batch)


class RCCDataLoader(DataLoader):

    def __init__(self, dataset, split, **kwargs):
        if split == 'train':
            kwargs['collate_fn'] = rcc_collate
        else:
            kwargs['collate_fn'] = rcc_collate_test

        super().__init__(dataset, **kwargs)
