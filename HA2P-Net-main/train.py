import os,shutil
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
from utils.dataloader import NucleiDataset
from trainer import Trainer
from torch.utils.data import DataLoader
import sys
import torch.nn as nn
from utils import prepare_sub_folder,get_config,collate_func
import torch
import numpy as np
import math
import argparse
import random
from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/consep_type_effv2s.yaml')
parser.add_argument('--name', type=str, default='effv2s_type')
parser.add_argument('--output_dir', type=str, default='myExperiments/consep')
parser.add_argument('--seed', type=int, default=888)
parser.add_argument('--logdir', type=str, default='logs/consep_1')
opts = parser.parse_args()

def check_manual_seed(seed):
    """ If manual seed is not specified, choose a
    random one and communicate it to the user.
    Args:
        seed: seed to check
    """
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print("Using manual seed: {seed}".format(seed=seed))
    return

if __name__ == '__main__':

    start_all_time = time.time()
    config=get_config(opts.config)
    train_dataset=NucleiDataset(config,opts.seed,is_train=True)
    check_manual_seed(opts.seed)
    train_loader=DataLoader(dataset=train_dataset, batch_size=config['train']['batch_size'], shuffle=True, drop_last=False, num_workers=config['train']['num_workers'],collate_fn=collate_func,pin_memory=True)

    output_directory = os.path.join(opts.output_dir, opts.name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config,os.path.join(output_directory,'config.yaml'))
#########
    train_directory = os.path.join(opts.logdir, 'train')
    if not os.path.exists(train_directory):
        print("Creating directory: {}".format(train_directory))
        os.makedirs(train_directory)
    writer_t = SummaryWriter(log_dir=train_directory)
#########
    trainer = Trainer(config)
    trainer.cuda()

    iter_per_epoch = len(train_loader)
    for epoch in range(config['train']['max_epoch']):
        ####
        print('Epoch: [{:d}/{:d}]'.format(epoch + 1, config['train']['max_epoch']))
        start_time = time.time()
        ####
        iteration = 1
        trainer.model.train()
        loss = 0
        for train_data in train_loader:

            for k in train_data.keys():
                if not isinstance(train_data[k], list):
                    train_data[k] = train_data[k].cuda().detach()
                else:
                    train_data[k] = [s.cuda().detach() if s is not None else s for s in train_data[k]]
            ins_loss, cate_loss, maskiou_loss= trainer.seg_updata_FMIX(train_data)
            # ins_loss, cate_loss, maskiou_loss,_,_ = trainer.seg_updata(train_data)
            sys.stdout.write(
                f'\r epoch:{epoch+1} step:{iteration}/{iter_per_epoch} ins_loss: {ins_loss} cate_loss: {cate_loss} maskiou_loss: {maskiou_loss}')
            iteration += 1

            loss += (ins_loss + cate_loss)
        loss /= iter_per_epoch
        writer_t.add_scalar('Loss', loss, epoch+1)

        trainer.scheduler.step()

        end_time = time.time()
        work_time = round((end_time - start_time), 4)
        print('\n train this epoch spend [{:.4f} s].loss{:.5f}'.format(work_time,loss))
        if (epoch+1) in [90,100,110,115,120]:
            trainer.save(checkpoint_directory, epoch)

    writer_t.close()
    end_all_time = time.time()
    work_all_time = end_all_time - start_all_time
    work_all_time = round(work_all_time, 3)
    work_minute_time = work_all_time / 60.0
    print('The time spent is [{:.3f} min].'.format(work_minute_time))
    # trainer.save(checkpoint_directory, epoch)

