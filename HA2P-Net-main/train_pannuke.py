from utils.dataloader import NucleiDataset,PannukeDataset
from trainer import Trainer
from torch.utils.data import DataLoader
import sys
from utils import prepare_sub_folder,get_config,collate_func
import torch
import numpy as np
import os,shutil
import argparse
import random
import time
from torch.utils.data import Subset
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/effv2s.yaml')
parser.add_argument('--name', type=str, default='pannuke')
parser.add_argument('--logdir', type=str, default='logs/pannuke')
parser.add_argument('--train_fold', type=int, default=1)
parser.add_argument('--val_fold', type=int, default=2)
parser.add_argument('--test_fold', type=int, default=3)
parser.add_argument('--output_dir', type=str, default='myExperiments/Pannuke')
parser.add_argument('--seed', type=int, default=888)


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
    # ia.random.seed(seed)

    print("Using manual seed: {seed}".format(seed=seed))
    return

def visualize(writer,loss,acc, precision, epoch):
    # 添加损失值到TensorBoard
    writer.add_scalar('Loss', loss, epoch)
    # 添加准确率值到TensorBoard
    writer.add_scalar('Accuracy', acc, epoch)
    writer.add_scalar('Precision', precision, epoch)


if __name__ == '__main__':
    start_all_time = time.time()
    config=get_config(opts.config)
    check_manual_seed(opts.seed)
    train_dataset=PannukeDataset(data_root=config['dataroot'], seed=opts.seed, is_train=True, fold=opts.train_fold,
                                 output_stride=config['model']['output_stride'])
    train_loader=DataLoader(dataset=train_dataset, batch_size=config['train']['batch_size'], shuffle=True,
                            drop_last=True, num_workers=config['train']['num_workers'],persistent_workers=True,
                            collate_fn=collate_func,pin_memory=True)
    print(f'Training processed data with size {len(train_dataset)}')
###
    val_dataset = PannukeDataset(data_root=config['dataroot'], seed=opts.seed, is_train=False, fold=opts.val_fold,
                               output_stride=config['model']['output_stride'])
    l = len(val_dataset)

    num_samples = int(l * 0.2)
    print(f'Validating processed data with size {num_samples}')

    subset = Subset(val_dataset, range(num_samples))
    val_loader = DataLoader(dataset=subset, batch_size=config['train']['batch_size'], shuffle=False,
                              drop_last=True, num_workers=config['train']['num_workers'], persistent_workers=True,
                               collate_fn=collate_func, pin_memory=True)
###
    output_directory = os.path.join(opts.output_dir, opts.name, 'train_{}_to_test_{}'.format( opts.train_fold,opts.test_fold))
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config,os.path.join(output_directory,'config.yaml'))

    trainer = Trainer(config)
    trainer.cuda()
###make logdir
    train_directory = os.path.join(opts.logdir,'train_{}_to_test_{}'.format( opts.train_fold,opts.test_fold), 'train')
    if not os.path.exists(train_directory):
        print("Creating directory: {}".format(train_directory))
        os.makedirs(train_directory)
    val_directory = os.path.join(opts.logdir, 'train_{}_to_test_{}'.format( opts.train_fold,opts.test_fold), 'val')
    if not os.path.exists(val_directory):
        print("Creating directory: {}".format(val_directory))
        os.makedirs(val_directory)
####make SummaryWriter obj
    writer_t = SummaryWriter(log_dir=train_directory)
    writer_v = SummaryWriter(log_dir=val_directory)
####
    # iteration=0
    iter_per_epoch=len(train_loader)
    iter_per_epoch_val = len(val_loader)
    min_loss = 1.05
    min_epoch = 0
    max_pre = 0.
    max_preepoch=0
    for epoch in range(config['train']['max_epoch']):
        ####
        print('Epoch: [{:d}/{:d}]'.format(epoch + 1, config['train']['max_epoch']))
        start_time = time.time()
        ####
        print('train:')
        train_loss = 0
        acc_t = 0
        pre_t = 0
        iteration = 1
        trainer.model.train()

        for train_data in train_loader:
            for k in train_data.keys():
                if not isinstance(train_data[k],list):
                    train_data[k]=train_data[k].cuda().detach()
                else:
                    train_data[k] = [s.cuda().detach() if s is not None else s for s in train_data[k]]
            ins_loss, cate_loss,maskiou_loss,at,pt=trainer.seg_updata(train_data)
            sys.stdout.write(f'\r epoch:{epoch+1} step:{iteration}/{iter_per_epoch} ins_loss: {ins_loss} cate_loss: {cate_loss} maskiou_loss: {maskiou_loss}')
            iteration+=1

            train_loss += (ins_loss+ cate_loss) ########
            acc_t += at
            pre_t += pt

        trainer.scheduler.step()

        end_time = time.time()
        work_time = round((end_time - start_time), 4)
        train_loss = train_loss / iter_per_epoch  #########
        acc_t /= iter_per_epoch
        pre_t /= iter_per_epoch
        visualize(writer_t, train_loss, acc_t, pre_t, epoch)
######val:
        print('\nval:')
        val_loss = 0
        acc = 0
        pre = 0
        iteration_val = 1
        trainer.model.eval()

        for val_data in val_loader:
            for k in val_data.keys():
                if not isinstance(val_data[k], list):
                    val_data[k] = val_data[k].cuda().detach()
                else:
                    val_data[k] = [s.cuda().detach() if s is not None else s for s in val_data[k]]
            # if (all(element is None for element in val_data['ins_labels'])):
            #     continue
            with torch.no_grad():
                val_ins_loss, val_cate_loss, val_maskiou_loss,a,p = trainer.seg_updata_val(val_data)
                acc += a
                pre += p
                val_loss += (val_ins_loss + val_cate_loss)

                sys.stdout.write(
                    f'\r val_epoch:{epoch + 1} step:{iteration_val}/{iter_per_epoch_val} ins_loss: {val_ins_loss} cate_loss: {val_cate_loss} maskiou_loss: {val_maskiou_loss}')
                iteration_val += 1
        ####
        val_loss = val_loss / iter_per_epoch_val
        acc /= iter_per_epoch_val
        pre /= iter_per_epoch_val

        if pre > max_pre:
            max_pre=pre
            max_preepoch=epoch+1
        if val_loss < min_loss and (epoch+1) >= 110:
            min_loss = val_loss
            min_epoch = epoch + 1
        visualize(writer_v, val_loss, acc, pre, epoch)
        print('\n train this epoch spend [{:.4f} s]. The val_loss: {:.6f}, val_accuracy: {:.6f}, val_precision:{:.6f}.'.format(work_time,val_loss,acc,pre))
        ###
        if val_loss == min_loss or (epoch+1) % 50 ==0 or ((epoch+1) in [90,110,115,120]):#(epoch+1)%50==0:((epoch+1)%10==0 and (epoch+1) > 200)
            trainer.save(checkpoint_directory, epoch)

    writer_v.close()
    writer_t.close()  # 关闭SummaryWriter对象

    end_all_time = time.time()
    work_all_time = end_all_time - start_all_time
    work_all_time = round(work_all_time, 3)
    work_minute_time = work_all_time / 60.0
    print('The time spent is [{:.3f} min]. Min average total loss is {:.6f} at epoch {}.Max pre {} at {}'.format(work_minute_time, min_loss,min_epoch,max_pre,max_preepoch))

