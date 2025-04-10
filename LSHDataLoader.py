from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random

class SigmoidDataset_eval(Dataset):
    def __init__(self, sequs=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], neg_ratio=1, desc_folder="./dataset_pre/desc", gt_folder="./dataset_pre/gt", eva_ratio=1) -> None:
        super().__init__()
        print(sequs)
        self.descs = []
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0
        for seq in sequs:
            desc_file = os.path.join(desc_folder, seq+'_desc.npz') # 
            gt_file = os.path.join(gt_folder, seq+'_gt.npz') #  gt
            self.descs.append(np.load(desc_file)["Descs"])
            gt = np.load(gt_file)
            pos = gt['pos'][-int(len(gt['pos'])*eva_ratio):]
            neg = gt['neg'][-int(len(gt['neg'])*eva_ratio):]
            self.gt_pos.append(pos)
            self.gt_neg.append(neg)
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)

    def __len__(self):
        # pos_num 
        # neg_num  
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, 0]

        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {"desc": self.descs[int(id_seq)][int(pair[0])], 'label': pair[1]*1.}

            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {"desc": self.descs[i-1][int(pair[0])], 'label': pair[1]*1.}
                return out


class SigmoidDataset_train(Dataset):
    def __init__(self, sequs=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], neg_ratio=1, desc_folder="./dataset_pre/desc", gt_folder="./dataset_pre/gt", eva_ratio=1) -> None:
        super().__init__()
        print(sequs)
        self.descs = []
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0
        for seq in sequs:
            desc_file = os.path.join(desc_folder, seq+'_desc.npz') # 
            gt_file = os.path.join(gt_folder, seq+'_gt.npz') #  gt
            self.descs.append(np.load(desc_file)["Descs"])
            gt = np.load(gt_file)
            pos = gt['pos'][:-int(len(gt['pos'])*eva_ratio)]
            neg = gt['neg'][:-int(len(gt['neg'])*eva_ratio)]
            self.gt_pos.append(pos)
            self.gt_neg.append(neg)
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio * self.pos_num)

    def __len__(self):
        return self.pos_num + self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, 0]

        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {"desc": self.descs[int(id_seq)][int(pair[0])], 'label': pair[1]*1.}

            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {"desc": self.descs[i-1][int(pair[0])], 'label': pair[1]*1.}
                return out


class evalDataset(Dataset):
    def __init__(self, sequs=['1', '2'], desc_folder="./dataset_pre/desc", gt_folder="./dataset_pre/gt") -> None:
        super().__init__()
        self.descs = []
        self.pairs = []
        self.dict = {}
        self.num = 0
        for iii, seq in enumerate(sequs):
            desc_file = os.path.join(desc_folder, seq+'_desc.npz') # 
            self.descs.append(np.load(desc_file)["Descs"])

            gt_file = os.path.join(gt_folder, seq+'_gt.npz') #  gt
            gt = np.load(gt_file)

            seq_pos = np.ones((gt['pos'].shape[0], 1)) * iii
            seq_neg = np.ones((gt['neg'].shape[0], 1)) * iii

            gt_pos = np.hstack((seq_pos, gt['pos']))
            gt_neg = np.hstack((seq_neg, gt['neg']))

            self.pairs.extend(gt_pos)
            self.pairs.extend(gt_neg)
            self.dict[iii] = seq

        self.num = len(self.pairs)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = self.pairs[idx]
        out = {"seq": self.dict[int(pair[0])], 'idx': int(pair[1]), "desc": self.descs[int(pair[0])][int(pair[1])], 'label': pair[2]}

        return out

