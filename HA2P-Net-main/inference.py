"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# from solo_trainer import solo_Trainer
from trainer import Trainer
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import scipy.io as scio

import torch
import os
import tifffile
import numpy as np
from PIL import Image
from utils import get_config,_imageshow,_imagesave
from skimage.util.shape import view_as_windows
from utils.imop import get_ins_info
from utils.metrics import get_fast_aji,get_fast_pq, get_dice_1,remap_label, get_cate_f1,run_nuclei_type_stat
from torchvision import transforms
import argparse
import cv2
from collections import Counter
import time
import scipy.io as scio

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='myExperiments/consep')
parser.add_argument('--save_dir', type=str, default='outPuts_result/consep') #only type consep
parser.add_argument('--name', type=str, default='effv2s_type')
parser.add_argument('--epoch',type=int,default=120)
parser.add_argument('--load_size',type=int,default=1024)
parser.add_argument('--patch_size',type=int,default=256)
parser.add_argument('--stride',type=int,default=128)
parser.add_argument('--use_type',type=bool,default=False)
opts = parser.parse_args()

if __name__ == '__main__':

    opts.config=os.path.join(opts.output_dir,'{}/config.yaml'.format(opts.name))
    config=get_config(opts.config)
    trainer = Trainer(config)
    trainer.cuda()

    load_size = opts.load_size
    patch_size = opts.patch_size
    stride = opts.stride

    state_path = os.path.join(opts.output_dir,'{}/checkpoints/model_{}.pt'.format(opts.name, '%04d' % (opts.epoch)))
    state_dict = torch.load(state_path)
    trainer.model.load_state_dict(state_dict['seg'])
    trainer.model.eval()
    if not config['image_norm_mean']:
        _mean = (0.5, 0.5, 0.5)
        _std = (0.5, 0.5, 0.5)
    else:
        _mean = tuple(config['image_norm_mean'])
        _std = tuple(config['image_norm_std'])
    im_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(_mean, _std)])

    f1 = {0:[],1:[],2:[],3:[],4:[]}
    mpq = {0:[],1:[],2:[],3:[],4:[]}

    ajis = []
    dices = []
    pqs = []
    dqs = []
    sqs = []

    time_cost = []

    root = os.path.join(config['dataroot'], 'test')
    stain_norm_type=config['stainnorm']

    test_dir_fp=os.path.join(root, 'Images') if stain_norm_type is None else os.path.join(root, f'Images_{stain_norm_type}')
    print(test_dir_fp,stain_norm_type,_mean,_std)
    test_img_fp=os.listdir(test_dir_fp)

    test_meta = [os.path.splitext(p)[0] for p in test_img_fp]
    test_img_fp=[os.path.join(test_dir_fp,f) for f in test_img_fp]
    for i, (test_fp, test_file_name) in enumerate(zip(test_img_fp,test_meta)):
        if 'tif' in test_fp:
            with tifffile.TiffFile(test_fp) as f:
                test_img = f.asarray()
        else:
            test_img = np.array(Image.open(test_fp))
        original_img=test_img.copy()#tifffile.TiffFile(os.path.join(root,'Images',f'{test_file_name}.tif')).asarray()#
        test_gt_path = os.path.join(root, 'Labels', test_file_name + '.mat')
        date_ = scio.loadmat(test_gt_path)

        test_GT = scio.loadmat(test_gt_path)['inst_map'].astype(np.int32)
        im_size = test_img.shape[0]

        assert patch_size % stride == 0 and load_size % patch_size == 0
        pad_size = (load_size - im_size + patch_size) // 2

        test_img = np.pad(test_img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
        crop_test_imgs = view_as_windows(test_img, (patch_size, patch_size, 3), (stride, stride, 3))[:, :, 0]

        pred_crop_test_imgs = []
        ins_num = 1
        output_seg = np.zeros((load_size + patch_size, load_size + patch_size), dtype=np.int32)
        score_list={}
        label_list={}

        time_c = 0

        for i in range(crop_test_imgs.shape[0]):
            for j in range(crop_test_imgs.shape[1]):
                crop_test_img = crop_test_imgs[i, j]
                crop_test_img = im_transform(crop_test_img).unsqueeze(0).cuda()
                start = time.time()
                with torch.no_grad():
                    output = trainer.prediction_single(crop_test_img,score_thr=0.4,update_thr=0.2)
                    # output = trainer.prediction_fast(crop_test_img)
                end = time.time()
                times =  end - start
                time_c += times
                if output is not None:
                    seg_masks, cate_labels, cate_scores = output
                else:
                    continue
                seg_masks = seg_masks.cpu().numpy()
                cate_labels = cate_labels.cpu().numpy()
                cate_scores = cate_scores.cpu().numpy()

                for ins_id in range(seg_masks.shape[0]):
                    seg_ = seg_masks[ins_id]
                    label_ = cate_labels[ins_id]
                    score_ = cate_scores[ins_id]
                    center_w, center_h, width, height = get_ins_info(seg_, method='bbox')
                    center_h = np.ceil(center_h)
                    center_w = np.ceil(center_w)
                    offset_h = i * stride
                    offset_w = j * stride
                    if center_h >= patch_size // 2 - stride // 2 and center_h <= patch_size // 2 + stride // 2 and center_w >= patch_size // 2 - stride // 2 and center_w <= patch_size // 2 + stride // 2:
                        focus_area = output_seg[offset_h:offset_h + patch_size, offset_w:offset_w + patch_size].copy()
                        if np.sum(np.logical_and(focus_area > 0, seg_)) == 0:
                            output_seg[offset_h:offset_h + patch_size, offset_w:offset_w + patch_size] = np.where(
                                focus_area > 0, focus_area, seg_ * (ins_num))#seg_ * (ins_num)
                            score_list[ins_num]=score_
                            label_list[ins_num] = label_
                            ins_num += 1
                        else:
                            compared_num, _ = Counter((focus_area * seg_).flatten()).most_common(2)[1]
                            assert compared_num > 0
                            compared_num = int(compared_num)
                            compared_score = score_list[compared_num]

                            if np.sum(np.logical_and(focus_area == compared_num, seg_)) / np.sum(
                                np.logical_or(focus_area == compared_num, seg_)) > 0.5:#IoU>0.1判断重叠
                                if compared_score > score_:
                                    pass
                                else:
                                    focus_area[focus_area==compared_num]=0
                                    output_seg[offset_h:offset_h + patch_size, offset_w:offset_w + patch_size]=focus_area
                                    output_seg[offset_h:offset_h + patch_size,
                                    offset_w:offset_w + patch_size] = np.where(
                                        focus_area > 0, focus_area, seg_ * (ins_num))
                                    score_list[ins_num] = score_
                                    label_list[ins_num] = label_
                                    ins_num += 1
                            else:
                                output_seg[offset_h:offset_h + patch_size, offset_w:offset_w + patch_size] = np.where(
                                    focus_area > 0, focus_area, seg_ * (ins_num))
                                score_list[ins_num] = score_
                                label_list[ins_num] = label_
                                ins_num += 1
        output_seg = output_seg[pad_size:-pad_size, pad_size:-pad_size]

        time_cost.append(time_c)

        for ui in np.unique(output_seg):
            if ui ==0:continue
            if np.sum(output_seg==ui)<16:
                output_seg[output_seg==ui]=0
##############
        print(20 * '-',f'{test_file_name}',20*'-')
        if opts.use_type:
            inst_type = []
            inst_centroid = []

            type_pred = np.copy(output_seg)
            type_pred.fill(0)
            for ui in np.unique(output_seg):
                if ui == 0:continue
                x_tmp = output_seg == ui
                center_w, center_h, width, height = get_ins_info(x_tmp*1, method='area')  #['bbox','circle','area']
                inst_type.append(label_list[ui])
                inst_centroid.append([center_w, center_h])
                type_pred += label_list[ui] * x_tmp

            pred = {'inst_map':output_seg,'inst_type':np.array(inst_type).reshape(len(inst_type), 1), 'inst_centroid':np.array(inst_centroid).reshape(len(inst_centroid), 2)}
            if not os.path.exists(f'{opts.save_dir}/{opts.name}'):
                os.makedirs(f'{opts.save_dir}/{opts.name}')
            scio.savemat(f'{opts.save_dir}/{opts.name}/{test_file_name}.mat', pred)

#######
        test_GT = remap_label(test_GT.copy())
        output_seg = remap_label(output_seg.copy(),ispred=True)
        aji = get_fast_aji(test_GT.copy(), output_seg.copy())
        dice = get_dice_1(test_GT.copy(), output_seg.copy())

        [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred] = get_fast_pq(test_GT.copy(), output_seg.copy(),match_iou=0.5)
        # print(f'dice {round(float(dice), 3)}|AJI {round(float(aji), 3)}|dq {round(float(dq), 3)}|sq {round(float(sq), 3)}|pq {round(float(pq), 3)}|gt up {len(unpaired_true)}|pred up {len(unpaired_pred)}')
        result_string = "dice {:.3f}|AJI {:.3f}|dq {:.3f}|sq {:.3f}|pq {:.3f}|gt up {}|pred up {}".format(
            round(float(dice), 3), round(float(aji), 3), round(float(dq), 3), round(float(sq), 3),
            round(float(pq), 3), len(unpaired_true), len(unpaired_pred)
        )
        print(result_string)

        title=f'DICE:{round(float(dice), 3)}, AJI:{round(float(aji), 3)},\n DQ:{round(float(dq), 3)}, SQ:{round(float(sq), 3)}, PQ:{round(float(pq), 3)}'
        # _imageshow(test_img,output_seg,test_GT,unpaired_pred,unpaired_true,title=title)
        if not os.path.exists(f'{opts.output_dir}/{opts.name}/images/pred'):
            os.makedirs(f'{opts.output_dir}/{opts.name}/images/pred')
        _imagesave(original_img,output_seg,None,f'{opts.output_dir}/{opts.name}/images/pred/{test_file_name}.png')


        dices.append(dice)
        ajis.append(aji)
        dqs.append(dq)
        sqs.append(sq)
        pqs.append(pq)
    print(20*'-'+'result'+20*'-')
    print(time_cost)
    print('FPS: {:.2f}'.format(sum(time_cost) / len(time_cost )))
    print(f'dice= {round(float(np.mean(dices)),3)}\tAJI= {round(float(np.mean(ajis)),3)}\tdq={round(float(np.mean(dqs)),3)}\tsq={round(float(np.mean(sqs)),3)}\tpq={round(float(np.mean(pqs)),3)}')
    if opts.use_type:
        pred_dir = f'{opts.save_dir}/{opts.name}/'
        true_dir = os.path.join(root, 'Labels')
        run_nuclei_type_stat(pred_dir, true_dir)