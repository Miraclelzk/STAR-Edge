import torch
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import numpy as np
from tqdm import tqdm
import time

import LocalSH.LocalSHFeature as localsh_feature

pc_dir = "/opt/data/private/LocalSphericalFeature/dataset/pc"
# pc_paths = natsorted([os.path.join(pc_dir, file) for file in os.listdir(pc_dir) if file.endswith('Descs_matrix.txt')])
pc_paths = natsorted([os.path.join(pc_dir, file) for file in os.listdir(pc_dir) if file.endswith('txt')], reverse=True)
# pc_paths = natsorted([os.path.join(pc_dir, file) for file in os.listdir(pc_dir) if file.endswith('txt')])
gt_dir = "/opt/data/private/LocalSphericalFeature/gt/"

save_dir = "/opt/data/private/LocalSphericalFeature/pre_data/"

if not os.path.exists(os.path.join(save_dir,"desc")):
    os.makedirs(os.path.join(save_dir,"desc"))

# LSC
bw = 10
kk = 26
sampleNum = bw * 4


for iii in range(len(pc_paths)):
    print(pc_paths[iii])
    pc_path = pc_paths[iii]
    pc_name = os.path.splitext(os.path.basename(pc_path))[0]

    pc_norm = np.loadtxt(pc_path)
    print(pc_norm.shape)
    pc = pc_norm[:, :3]
    norm = pc_norm[:, -3:]
    start_time = time.time() 

    result = localsh_feature.ComLSHF_knn_upsample(pc, bw, kk, sampleNum)

    end_time = time.time()  #
    execution_time = end_time - start_time  #

    print(f: {execution_time}")

    # desc = result["Descs"]

    np.savez(save_dir + f"desc/{pc_name}_desc.npz", **result)


