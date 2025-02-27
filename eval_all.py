import torch
from net import *
from LSHDataLoader import evalDataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn import metrics
from matplotlib import pyplot as plt
import sys
import yaml
import argparse
from datetime import date
import datetime
from net import *
from natsort import natsorted
import time

import LocalSH.LocalSHFeature as localsh_feature
import LocalSH.NormalRefine as normal_refine

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_ply_cls(save_file, pcff_np):

    ply_header = f'''ply
    format ascii 1.0
    element vertex {len(pcff_np)}
    property float x
    property float y
    property float z
    property float nx
    property float ny
    property float nz
    property float scale1
    property float scale2
    end_header
    '''

    with open(save_file, 'w') as f:
        f.write(ply_header)
        for i in range(len(pcff_np)):
            x, y, z, nx, ny, nz, scale1, scale2  = pcff_np[i]
            f.write(f'{x} {y} {z} {nx} {ny} {nz} {scale1} {scale2}\n')

def save_ply_edge(save_file, pcff_np):

    ply_header = f'''ply
    format ascii 1.0
    element vertex {len(pcff_np)}
    property float x
    property float y
    property float z
    end_header
    '''
    with open(save_file, 'w') as f:
        f.write(ply_header)
        for i in range(len(pcff_np)):
            x, y, z = pcff_np[i]
            f.write(f'{x} {y} {z} \n')

def eval(seqs, model_file, desc_folder, gt_folder, save_folder):
    net = DescClassifier()
    net.load(model_file)

    test_dataset = evalDataset(seqs, desc_folder, gt_folder)

    net.to(device=device)
    net.eval()
    testdataloader = DataLoader(dataset = test_dataset, batch_size=16384, shuffle=False, num_workers=12)
    
    seq = []
    idx = []
    pred = []
    gt = []

    
    with torch.no_grad():
        for i_batch, sample_batch in tqdm(enumerate(testdataloader), total=len(testdataloader), desc="Eval seq "):
            out= net(sample_batch["desc"].to(device=device).float())
            outlabel = out.cpu().tolist()
            label = sample_batch['label']
            seq_ = sample_batch['seq']
            idx_ = sample_batch['idx']

            seq.extend(seq_)
            idx.extend(idx_.tolist())
            pred.extend(outlabel)
            gt.extend(label.tolist())
    pred = np.nan_to_num(pred)

    uniq = np.unique(seq)
    str_groups = {str: np.where(np.array(seq) == str)[0] for str in uniq}

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    for seq_str, ind in str_groups.items():
        print(seq_str)

        save_db = np.array([np.array(idx)[ind], np.array(pred)[ind], np.array(gt)[ind]])
        save_db = save_db.T
        np.savetxt(os.path.join(save_folder, seq_str+'_cls.txt'), save_db, fmt = "%d %f %d")
        np.save(os.path.join(save_folder, seq_str + '_cls.npy'), save_db)


pc_dir = "demo/data"
# pc_paths = natsorted([os.path.join(pc_dir, file) for file in os.listdir(pc_dir) if file.endswith('Descs_matrix.txt')])
pc_paths = natsorted([os.path.join(pc_dir, file) for file in os.listdir(pc_dir) if file.endswith('txt')])
print(pc_paths)
# pc_paths = natsorted([os.path.join(pc_dir, file) for file in os.listdir(pc_dir) if file.endswith('txt')])
# gt_dir = "/opt/data/private/LocalSphericalFeature/gt/"

save_dir = "demo/result/"

model_file ="model/best.ckpt"
# 
bw = 10
kk = 26
sampleNum = bw * 4


batch_size = 2048

is_normalRefine = True


if __name__ == '__main__':

    net = DescClassifier()
    net.load(model_file)
    net.to(device=device)
    net.eval()

    for iii in range(len(pc_paths)):
        print(pc_paths[iii])

        pc_path = pc_paths[iii]
        pc_name = os.path.splitext(os.path.basename(pc_path))[0]
        pc_norm = np.loadtxt(pc_path)
        print(pc_norm.shape)
        pc = pc_norm[:, :3]
        norm = pc_norm[:, -3:]

        desc_path = os.path.join(save_dir , f"{pc_name}_desc.npz")
        if not os.path.isfile(desc_path):
            
            start_time = time.time() 
            result = localsh_feature.ComLSHF_knn_upsample(pc, bw, kk, sampleNum)
            execution_time1 = end_time = time.time() - start_time  
            print(f"Descs runing time: {execution_time1} s")

            Descs = result["Descs"]
            normals = result["normals"]
            neighboor = result["neighboor"]

            np.savez(os.path.join(save_dir , f"{pc_name}_desc.npz"), **result)
        else:
            result = np.load(desc_path)

            Descs = result["Descs"]
            normals = result["normals"]
            neighboor = result["neighboor"].tolist()

        n = Descs.shape[0]
        Descs = torch.tensor(Descs, device='cuda')

        pred = []

        with torch.no_grad():
            for start_idx in tqdm(range(0, n, batch_size)):

                end_idx = min(start_idx + batch_size, n)
                Descs_batch = Descs[start_idx:end_idx]
                out= net(Descs_batch.to(device=device).float())
                outlabel = out.cpu().tolist()

                pred.extend(outlabel)
        
        flag = np.array([1 if p > 0.5 else 0 for p in pred], dtype=int)

        print(flag.shape)
        
        if is_normalRefine:
            new_norm = normal_refine.checkNormalDirection(norm, normals)

            mu =0.1

            start_time = time.time() 
            NewEdgePoints = normal_refine.EdgePointRefine(flag, neighboor, pc, new_norm, mu)
            execution_time2 = end_time = time.time() - start_time  
            print(f"Refine runing time: {execution_time2} s")

            save_ply_edge(os.path.join(save_dir, pc_name+'_edge.ply'), NewEdgePoints)
        else:

            mu =0.1

            start_time = time.time() 
            NewEdgePoints = normal_refine.EdgePointRefine(flag, neighboor, pc, norm, mu)
            execution_time2 = end_time = time.time() - start_time  
            print(f"Refine runing time: {execution_time2} s")

            save_ply_edge(os.path.join(save_dir, pc_name+'_edge_NoRefine.ply'), NewEdgePoints)

        if not os.path.isfile(desc_path):
            print(f"{pc_name} runing time: {execution_time1 + execution_time2} s")







