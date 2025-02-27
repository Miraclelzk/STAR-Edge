import string
import torch
from net import *
from LSHDataLoader import SigmoidDataset_train, SigmoidDataset_eval
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import os
import argparse
import yaml
import sys

from torch.utils.tensorboard.writer import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from datetime import date

torch.manual_seed(5)

def train(cfg):

    day_name = date.today()
    comment=f'bath_size={cfg.batch_size} lr={cfg.learning_rate} weight_decay={cfg.weight_decay} eval_ratio={cfg.eval_ratio} neg_ratio={cfg.neg_ratio}'
    logs_name = str(day_name) + comment
    writer = SummaryWriter(log_dir = "runs/" + str(day_name) + "/" + logs_name)
    ## model ##
    net = DescClassifier() 
    net.to(device=device)
    print(net)
    sequs = cfg.all_seqs
    # for s in cfg.test_seqs:
    #     sequs.remove(s)

    sequs_train = []
    for seq in sequs:
        for s in cfg.sample:
            sequs_train.append(f"{seq}_sample{s}")
        for n in cfg.noise:
            sequs_train.append(f"{seq}_sample0.5_noise{n}")

    sequs_test = []
    for seq in cfg.test_seqs:
        for s in cfg.sample:
            sequs_test.append(f"{seq}_sample{s}")
        for n in cfg.noise:
            sequs_test.append(f"{seq}_sample0.5_noise{n}")
    
    train_dataset = SigmoidDataset_train(sequs=sequs_train, neg_ratio=cfg.neg_ratio, eva_ratio=cfg.eval_ratio, desc_folder=cfg.desc_folder, gt_folder=cfg.gt_folder)
    test_dataset = SigmoidDataset_eval(sequs=sequs_test, neg_ratio=cfg.neg_ratio, eva_ratio=cfg.eval_ratio, desc_folder=cfg.desc_folder, gt_folder=cfg.gt_folder)

    batch_size = cfg.batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    epoch = cfg.max_epoch
    starting_epoch = 0
    batch_num = 0

    #
    if not cfg.model == "":
        checkpoint = torch.load(cfg.model)
        starting_epoch = checkpoint['epoch']
        batch_num = checkpoint['batch_num']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    temp_f1 = 0
    for i in range(starting_epoch, epoch):
        print("epoch: ",i)
        net.train()
        pred = []
        gt = []
        for i_batch, sample_batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train epoch '+str(i), leave=False):
            optimizer.zero_grad()
            out= net(sample_batch["desc"].to(device=device).float())
            labels = sample_batch["label"].to(device=device)
            #（binary cross-entropy loss）
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, labels)

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                writer.add_scalar('total loss', loss.cpu().item(), global_step=batch_num)
                writer.add_scalar('loss', loss.cpu().item(), global_step=batch_num)

                batch_num += 1
                outlabel = out.cpu().numpy()
                label = sample_batch['label'].cpu().numpy()
                mask = (label > 0.9906840407) | (label < 0.0012710163)
                label = label[mask]
                label[label < 0.5] = 0
                label[label > 0.5] = 1
                pred.extend(outlabel[mask].tolist())
                gt.extend(label.tolist())
                
        pred = np.array(pred, dtype='float32')
        pred = np.nan_to_num(pred)
        gt = np.array(gt, dtype='float32')
        precision, recall, _ = metrics.precision_recall_curve(gt, pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        trainaccur = np.max(F1_score)
        print('Train F1:', trainaccur)
        writer.add_scalar('train f1', trainaccur, global_step=i)
        lastaccur = test(net=net, dataloader=test_loader)
        writer.add_scalar('eval f1', lastaccur, global_step=i)
        print('Eval F1:', lastaccur)
        torch.save({'epoch': i, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'batch_num': batch_num}, os.path.join(cfg.log_dir, cfg.env, str(i)+'.ckpt'))
        if(lastaccur > temp_f1):
            temp_f1 = lastaccur
            torch.save({'epoch': i, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'batch_num': batch_num}, os.path.join(cfg.log_dir, cfg.env, 'best.ckpt'))


def test(net, dataloader):
    net.eval()
    pred = []
    gt = []
    with torch.no_grad():
        for i_batch, sample_batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Eval", leave=False):
            out= net(sample_batch["desc"].to(device=device).float())
            out = out.cpu()
            outlabel = out
            label = sample_batch['label']
            mask = (label > 0.9906840407) | (label < 0.0012710163)
            label = label[mask]
            label[label < 0.5] = 0
            label[label > 0.5] = 1
            pred.extend(outlabel[mask])
            gt.extend(label)
        pred = np.array(pred, dtype='float32')
        gt = np.array(gt, dtype='float32')
        pred = np.nan_to_num(pred)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt, pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        testaccur = np.max(F1_score)
        return testaccur

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='log/',
                        help='Log dir. [default: log]')
    parser.add_argument('--max_epoch', type=int, default=40,
                        help='Epoch to run. [default: 20]')
    parser.add_argument('--run', default='1',
                        help='running. [default: running]')

    cfg = parser.parse_args()

    #
    running_id = 'running2_test'
    print(running_id)

    config_filename ='./config/config_thin.yaml'

    with open(config_filename, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    cfg.env = config[running_id]['env']
    cfg.all_seqs = config[running_id]['all_seqs']
    cfg.test_seqs = config[running_id]['test_seqs']
    cfg.sample = config[running_id]['sample']
    cfg.noise = config[running_id]['noise']

    cfg.desc_folder = config[running_id]['desc_folder']
    cfg.gt_folder = config[running_id]['gt_folder']

    cfg.learning_rate = config[running_id]['learning_rate']
    cfg.weight_decay = config[running_id]['weight_decay']
    cfg.eval_ratio = config[running_id]['eval_ratio']
    cfg.neg_ratio = config[running_id]['neg_ratio']
    cfg.batch_size = config[running_id]['batch_size']

    if(not os.path.exists(os.path.join(cfg.log_dir, cfg.env))):
        os.makedirs(os.path.join(cfg.log_dir, cfg.env))

    train(cfg)
