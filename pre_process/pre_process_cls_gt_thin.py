import torch
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import numpy as np
from tqdm import tqdm

pc_dir = "/opt/data/private/LocalSphericalFeature/dataset/pc"
# pc_paths = natsorted([os.path.join(pc_dir, file) for file in os.listdir(pc_dir) if file.endswith('Descs_matrix.txt')])
pc_paths = natsorted([os.path.join(pc_dir, file) for file in os.listdir(pc_dir) if file.endswith('txt')])
gt_dir = "/opt/data/private/LocalSphericalFeature/dataset/gt/"

save_dir = "/opt/data/private/LocalSphericalFeature/pre_data/"
if not os.path.exists(os.path.join(save_dir,"pc_normal")):
    os.makedirs(os.path.join(save_dir,"pc_normal"))
if not os.path.exists(os.path.join(save_dir,"cls_gt")):
    os.makedirs(os.path.join(save_dir,"cls_gt"))


#0.32
dist_threshold =0.3  # gt

batch_size = 100

for iii in range(len(pc_paths)):
    print(pc_paths[iii])
    pc_path = pc_paths[iii]
    pc_name = os.path.splitext(os.path.basename(pc_path))[0]
    pc_id = pc_name.split('_')[0]

    gt = np.loadtxt(gt_dir + pc_id + "_gt.txt")
    print(gt.shape)

    pc_norm = np.loadtxt(pc_path)
    print(pc_norm.shape)
    pc = pc_norm[:, :3]
    norm = pc_norm[:, -3:]

    id_pos = []
    id_neg = []

    n = pc.shape[0]
    gt = torch.tensor(gt, device='cuda')
    pc = torch.tensor(pc, device='cuda')

    for start_idx in tqdm(range(0, n, batch_size)):
        end_idx = min(start_idx + batch_size, n)
        pc_batch = pc[start_idx:end_idx]
        dis = torch.norm(pc_batch[:, None, :] - gt[None, :, :], dim=-1)

        min_dis = torch.min(dis, dim=1).values
        
        batch_id_pos = torch.where(min_dis < dist_threshold)[0]
        batch_id_neg = torch.where(min_dis > dist_threshold)[0]

        global_id_pos = batch_id_pos + start_idx
        global_id_neg = batch_id_neg + start_idx
        id_pos.extend(global_id_pos.cpu().numpy())
        id_neg.extend(global_id_neg.cpu().numpy())

    id_pos = np.array(id_pos).reshape(-1, 1)
    id_neg = np.array(id_neg).reshape(-1, 1)

    id_pos = np.concatenate([id_pos, (id_pos[:]*0+1).reshape(-1, 1)], axis=1)
    id_neg = np.concatenate([id_neg, (id_neg[:]*0).reshape(-1, 1)], axis=1)
    print(id_pos.shape)
    print(id_neg.shape)

    np.save(save_dir + f"pc_normal/{pc_name}.npy", pc_norm)
    np.savez(save_dir + f"cls_gt/{pc_name}_gt.npz", pos=id_pos, neg=id_neg)
