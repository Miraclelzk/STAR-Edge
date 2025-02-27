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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='1',
                        help='running. [default: running]')
    cfg = parser.parse_args()

    running_id = 'running2_test'
    print(running_id)

    config_filename ='config/config_thin.yaml'
    with open(config_filename, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    learning_rate = config[running_id]['learning_rate']
    weight_decay = config[running_id]['weight_decay']
    eval_ratio = config[running_id]['eval_ratio']
    neg_ratio = config[running_id]['neg_ratio']
    batch_size = config[running_id]['batch_size']

    cfg.test_seqs = config[running_id]['test_seqs']
    cfg.sample = config[running_id]['sample']
    cfg.noise = config[running_id]['noise']

    cfg.pc_folder = config[running_id]['pc_folder']
    cfg.desc_folder = config[running_id]['desc_folder']
    cfg.gt_folder = config[running_id]['gt_folder']
    cfg.save_folder = config[running_id]['save_folder']

    env_ = config[running_id]['env']
    cfg.env_ = '{}'.format(env_)

    
    sequs_test = []
    for seq in cfg.test_seqs:
        for s in cfg.sample:
            sequs_test.append(f"{seq}_sample{s}")
        for n in cfg.noise:
            sequs_test.append(f"{seq}_sample0.5_noise{n}")

    save_cls_folder = os.path.join(cfg.save_folder, "cls")
    if not os.path.exists(save_cls_folder):
        os.makedirs(save_cls_folder)

    eval(seqs=sequs_test, model_file=cfg.model, desc_folder=cfg.desc_folder, gt_folder=cfg.gt_folder , save_folder = save_cls_folder)

    save_cls_ply_folder = os.path.join(cfg.save_folder, "ply_cls")
    if not os.path.exists(save_cls_ply_folder):
        os.makedirs(save_cls_ply_folder)

    for name in sequs_test:
        print(name)

        cls_file = np.load(os.path.join(save_cls_folder, name+'_cls.npy'))

        dict_pred = {}
        dict_gt = {}


        for row in cls_file:

            key = int(row[0] )
            value1 = float(1.0 if row[1] > 0.5 else 0.0)
            value2 = float(row[2]  )

            dict_pred[key] = value1
            dict_gt[key] = value2

        desc_file = np.load(os.path.join(cfg.desc_folder, name+'_desc.npz')) 
        Descs = desc_file["Descs"]
        normals = desc_file["normals"]
        neighboor = desc_file["neighboor"].tolist()
        
        pc_file = np.load(os.path.join(cfg.pc_folder, name+'.npy'))
        pc = pc_file[:, :3]
        norm = pc_file[:, -3:]

        new_norm = normal_refine.checkNormalDirection(norm, normals)

        pc_out = np.full((pc.shape[0], 8), -1.0)
        pc_out[:, :3] = pc
        pc_out[:, 3:6] = new_norm
        for key in dict_pred.keys():
            pc_out[key, 6] = dict_pred[key]
            pc_out[key, 7] = dict_gt[key]


        save_ply_cls(os.path.join(save_cls_ply_folder, name+'_cls.ply'), pc_out)

