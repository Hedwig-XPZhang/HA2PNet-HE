# @auther:guopeng
# @time:2023/9/5 9:45
# @file:pic_show.py
# @description:
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.segmentation import find_boundaries
import os, glob
from tqdm import trange, tqdm
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 600

data_dir = 'datasets/PanNuKe'  # location to extracted folds
output_dir = '/usr/local/data/guopeng/show_Folds/fold3_base13'  # location to save op data

# os.chdir(data_dir)
# folds = os.listdir(data_dir)


def get_boundaries(raw_mask):
    '''
    for extracting instance boundaries form the goundtruth file
    '''
    bdr = np.zeros(shape=raw_mask.shape)
    for i in range(raw_mask.shape[-1] - 1):  # because last chnnel is background
        bdr[:, :, i] = find_boundaries(raw_mask[:, :, i], connectivity=1, mode='thick', background=0)
    bdr = np.sum(bdr, axis=-1)
    return bdr.astype(np.uint8)


# for i, j in enumerate(folds):
j = 3
# get paths
print('Loading Data for fold{}, Wait...'.format(j))
img_path = data_dir + '/images/fold{}/images.npy'.format(j)
type_path = data_dir + '/images/fold{}/types.npy'.format(j)
mask_path = './outputs/pannuke_exp/train_1_to_test_3base/masks.npy'##'./outputs/pannuke_exp_effv2s_my/train_1_to_test_3/masks.npy'#data_dir + '/masks/fold{}/masks.npy'.format(j)
print(40 * '=', '\n', j, 'Start\n', 40 * '=')

# laod numpy files
masks = np.load(file=mask_path, mmap_mode='r')  # read_only mode
images = np.load(file=img_path, mmap_mode='r')  # read_only mode
types = np.load(file=type_path)

# creat directories to save images
try:
    #os.mkdir(output_dir)
    os.mkdir(output_dir)
    os.mkdir(output_dir + '/images')
    os.mkdir(output_dir + '/sem_masks')
    os.mkdir(output_dir + '/inst_masks')
    os.mkdir(output_dir + '/overlay')
except FileExistsError:
    pass
########################################################
import torch
from utils.imop import get_ins_info,gaussian_radius,draw_gaussian

def process_label(gt_labels_raw, gt_masks_raw, iou_threshold=0.3, tau=0.5):
    w, h = 256, 256
    cate_label = np.zeros([5, 64, 64], dtype=np.float64)
    ins_label = np.zeros([64 ** 2, w, h], dtype=np.int16)
    ins_ind_label = np.zeros([64 ** 2], dtype=np.bool_)
    if gt_masks_raw is not None:
        gt_labels = gt_labels_raw
        gt_masks = gt_masks_raw
        for seg_mask, gt_label in zip(gt_masks, gt_labels):
            center_w, center_h, width, height = get_ins_info(seg_mask, method='bbox')
            radius = max(gaussian_radius((width, height), iou_threshold), 0)
            coord_h = int((center_h / h) / (1. / 64))
            coord_w = int((center_w / w) / (1. / 64))
            temp = draw_gaussian(cate_label[gt_label - 1], (coord_w, coord_h), (radius / 4))
            non_zeros = (temp > tau).nonzero()
            label = non_zeros[0] *64 + non_zeros[1]  # label = int(coord_h * grid_size + coord_w)#
            cate_label[gt_label - 1, coord_h, coord_w] = 1
            label = int(coord_h * 64 + coord_w)  #
            ins_label[label, :, :] = seg_mask
            ins_ind_label[label] = True
    ins_label = np.stack(ins_label[ins_ind_label], 0)
    return cate_label, ins_label, ins_ind_label

def show_heat(tmask):
    olabel={}
    omask=np.zeros((256,256),dtype=np.int16)
    for p in range(5):
        ids=np.unique(tmask[:,:,p])
        if len(ids) ==1:
            continue
        else:
            for id in ids:
                if id==0:continue
                omask[tmask[:,:,p]==id]=id
                olabel[id]=p+1

    mask = omask
    label_dic = olabel

    cate_labels=[]
    ins_labels=[]
    for e, ui in enumerate(np.unique(mask)):
        if ui ==0:
            assert e==ui
            continue
        tmp_mask=mask==ui
        label=label_dic[ui]
        ins_labels.append(((tmp_mask)*1).astype(np.int32))
        cate_labels.append(label)

    if len(cate_labels)>0:
        cate_labels, ins_labels, ins_ind_labels=process_label(gt_labels_raw=np.array(cate_labels), gt_masks_raw=np.array(ins_labels),)
        cate_labels=torch.from_numpy(np.array(cate_labels)).float()
    else:
        cate_labels=torch.from_numpy(np.zeros([5,64,64])).float()

    result = np.zeros((64, 64))
    for i in range(5):
        label_ = cate_labels[i]
        label_ = np.where(label_ == 1, i+1, 0)
        result += label_

    # 自定义颜色映射
    colors = ['black', 'red','green','blue' ,'yellow', 'orange']  # 定义 1 到 5 对应的颜色
    # 创建自定义颜色映射
    cmap = plt.cm.colors.ListedColormap(colors)

    fig, axes = plt.subplots(3,2)
    a = 0
    for i in range(3):
        for j in range(2):
            axes[i, j].imshow(cate_labels[a], cmap='jet', interpolation='nearest', vmin=0, vmax=1)
            axes[i, j].set_title('Class {}'.format(a + 1))
            axes[i, j].axis('off')
            a+=1
            if a == 5:
                axes[2, 1].imshow(result, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
                axes[2, 1].set_title('local peak')
                axes[2, 1].axis('off')
                break
    plt.show()

#####################################################################
for k in trange(images.shape[0], desc='Writing files for fold{}'.format(j), total=images.shape[0]):
    raw_image = images[k, :, :, :].astype(np.uint8)
    raw_mask = masks[k, :, :, :]

    sem_mask = np.argmax(raw_mask, axis=-1).astype(np.uint8)

    # swaping channels 0 and 5 so that BG is at 0th channel
    sem_mask = np.where(sem_mask == 5, 6, sem_mask)
    sem_mask = np.where(sem_mask == 0, 5, sem_mask)
    sem_mask = np.where(sem_mask == 6, 0, sem_mask)

    tissue_type = types[k]
    instances = get_boundaries(raw_mask)

    # show_heat(masks[k])
    # # for plotting it'll slow down the process considerabelly
    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(instances)
    # ax[1].imshow(sem_mask)
    # ax[2].imshow(raw_image)
    # plt.show()

    unique_values = np.unique(sem_mask).tolist()
    if 0 in unique_values:
        unique_values.remove(0)

    # unique_values.remove(0)##
    color_set = [(0,255,0),(0,0,255),(255,255,0),(255,125,0),(255,0,0)]

    image_ = raw_image.copy()
    for i in unique_values:
        color_map = ((sem_mask == i) * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(color_map.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        overlay = cv2.drawContours(image_, contours, -1, tuple(color_set[i-1]), 2)

    # fig, ax = plt.subplots(2, 2)
    # ax[0][0].imshow(instances)
    # ax[0,0].set_title('instances')
    # ax[0,0].axis('off')
    # ax[1][0].imshow(sem_mask)
    # ax[1, 0].set_title('mask')
    # ax[1, 0].axis('off')
    # ax[0][1].imshow(raw_image)
    # ax[0, 1].set_title('image')
    # ax[0, 1].axis('off')
    # ax[1][1].imshow(overlay)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    # save file in op dir
    Image.fromarray(sem_mask).save(
        output_dir + '/sem_masks/sem_{}_{}_{:05d}.png'.format(tissue_type, j, k))
    Image.fromarray(instances).save(
        output_dir + '/inst_masks/inst_{}_{}_{:05d}.png'.format(tissue_type, j, k))
    Image.fromarray(raw_image).save(output_dir + '/images/img_{}_{}_{:05d}.png'.format(tissue_type, j, k))
    Image.fromarray(overlay).save(output_dir + '/overlay/over_{}_{}_{:05d}.png'.format(tissue_type, j, k))
    # if k == 5:
    #     break























