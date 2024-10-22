import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='pannuke_hrnet64_gam')
parser.add_argument('--save_path', type=str, default='outPuts_result/pannuke_hrnet64_gam')
parser.add_argument('--train_fold', type=int, default=1)
parser.add_argument('--test_fold', type=int, default=3)
opts = parser.parse_args()


import os
import numpy as np
import pandas as pd
from utils_eval import get_fast_pq, remap_label, binarize

tissue_types = [
                'Adrenal_gland',
                'Bile-duct',
                'Bladder',
                'Breast',
                'Cervix',
                'Colon',
                'Esophagus',
                'HeadNeck',
                'Kidney',
                'Liver',
                'Lung',
                'Ovarian',
                'Pancreatic',
                'Prostate',
                'Skin',
                'Stomach',
                'Testis',
                'Thyroid',
                'Uterus'
                ]

def main():

    pred_p = rf'outputs/{opts.name}/train_{opts.train_fold}_to_test_{opts.test_fold}'
    true_p = rf'datasets/PanNuKe/masks/fold{opts.test_fold}'
    type_p = rf'datasets/PanNuKe/images/fold{opts.test_fold}'
    save_path = opts.save_path
    # true_root = args['--true_path']
    # pred_root = args['--pred_path']
    # save_path = args['--save_path']

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    true_path = os.path.join(true_p,'masks.npy')  # path to the GT for a specific split
    pred_path = os.path.join(pred_p, 'masks.npy')  # path to the predictions for a specific split
    types_path = os.path.join(type_p,'types.npy') # path to the nuclei types

    # load the data
    true = np.load(true_path)
    pred = np.load(pred_path)
    types = np.load(types_path)

    print(true.shape)
    print('************')
    print(pred.shape)

    # from sklearn.metrics import precision_recall_fscore_support
    # from tabulate import tabulate
    # for a in range(5):
    #     ground_truth_labels = true[:, :, :, a].ravel()
    #     score_value = pred[:, :, :, a].ravel()
    #
    #     ground_truth_labels[ground_truth_labels > 0] = 1
    #     score_value[score_value > 0] = 1
    #
    #     pre, rec, f1, _ = precision_recall_fscore_support(ground_truth_labels, score_value, average='macro')
    #     print(
    #         tabulate([[a + 1, pre, rec, f1]], headers=['class', 'Precision', 'Recall', 'F1-Score'], tablefmt='orgtbl'))




    mPQ_all = []
    bPQ_all = []
    p_all = []
    r_all = []
    f1_all = []
    p_d = []
    r_d = []
    f1_d = []
    # loop over the images
    for i in range(true.shape[0]):
        pq = []
        p_each = []
        r_each = []
        f1_each = []

        true_bin = binarize(true[i,:,:,:5], False)
        pred_bin = binarize(pred[i,:,:,:5], True)

        if len(np.unique(true_bin)) == 1:
            pq_bin = np.nan # if ground truth is empty for that class, skip from calculation
            pb,rb,f1b = np.nan,np.nan,np.nan
        else:
            [_, _, pq_bin], _ ,[pb,rb,f1b]= get_fast_pq(true_bin, pred_bin, match_iou=0.5) # compute PQ

        # loop over the classes
        for j in range(5):
            pred_tmp = pred[i,:,:,j]
            pred_tmp = pred_tmp.astype('int32')
            true_tmp = true[i,:,:,j]
            true_tmp = true_tmp.astype('int32')
            pred_tmp = remap_label(pred_tmp, ispred=True)
            true_tmp = remap_label(true_tmp)

            if len(np.unique(true_tmp)) == 1:
                pq_tmp = np.nan # if ground truth is empty for that class, skip from calculation
                p, r , f1 = np.nan, np.nan,np.nan
            else:
                [_, _, pq_tmp] , _, [p,r,f1] = get_fast_pq(true_tmp, pred_tmp, match_iou=0.5) # compute PQ

            pq.append(pq_tmp)

            # p_each.append(p)
            # r_each.append(r)
            # f1_each.append(f1)

        mPQ_all.append(pq)
        bPQ_all.append([pq_bin])
        # p_all.append(p_each)
        # r_all.append(r_each)
        # f1_all.append(f1_each)
        # p_d.append(pb)
        # r_d.append(rb)
        # f1_d.append(f1b)
    # using np.nanmean skips values with nan from the mean calculation
    mPQ_each_image = [np.nanmean(pq) for pq in mPQ_all]
    bPQ_each_image = [np.nanmean(pq_bin) for pq_bin in bPQ_all]

    # class metric
    neo_PQ = np.nanmean([pq[0] for pq in mPQ_all])
    inflam_PQ = np.nanmean([pq[1] for pq in mPQ_all])
    conn_PQ = np.nanmean([pq[2] for pq in mPQ_all])
    dead_PQ = np.nanmean([pq[3] for pq in mPQ_all])
    nonneo_PQ = np.nanmean([pq[4] for pq in mPQ_all])

    # for a in range(5):
    #     p_ = np.nanmean([p_each[a] for p_each in p_all])
    #     r_ = np.nanmean([r_each[a] for r_each in r_all])
    #     f1_ = np.nanmean([f1_each[a] for f1_each in f1_all])
    #     print(a,'>>>','P=',p_,'r=', r_,'f1=',f1_)
    # pbd=np.nanmean(p_d)
    # rbd = np.nanmean(r_d)
    # f1bd = np.nanmean(f1_d)
    # print('detect', '>>>', 'P=', pbd, 'r=', rbd, 'f1=', f1bd)

    # Print for each class
    print('Printing calculated metrics on a single split')
    print('-'*40)
    print('Neoplastic PQ: {}'.format(neo_PQ))
    print('Inflammatory PQ: {}'.format(inflam_PQ))
    print('Connective PQ: {}'.format(conn_PQ))
    print('Dead PQ: {}'.format(dead_PQ))
    print('Non-Neoplastic PQ: {}'.format(nonneo_PQ))
    print('-' * 40)

    # Save per-class metrics as a csv file
    for_dataframe = {'Class Name': ['Neoplastic', 'Inflam', 'Connective', 'Dead', 'Non-Neoplastic'],
                        'PQ': [neo_PQ, inflam_PQ, conn_PQ, dead_PQ, nonneo_PQ]}
    df = pd.DataFrame(for_dataframe, columns=['Class Name', 'PQ'])
    df.to_csv(save_path + '/class_stats.csv')

    # Print for each tissue
    all_tissue_mPQ = []
    all_tissue_bPQ = []
    for tissue_name in tissue_types:
        indices = [i for i, x in enumerate(types) if x == tissue_name]
        tissue_PQ = [mPQ_each_image[i] for i in indices]
        print('{} PQ: {} '.format(tissue_name, np.nanmean(tissue_PQ)))
        tissue_PQ_bin = [bPQ_each_image[i] for i in indices]
        print('{} PQ binary: {} '.format(tissue_name, np.nanmean(tissue_PQ_bin)))
        all_tissue_mPQ.append(np.nanmean(tissue_PQ))
        all_tissue_bPQ.append(np.nanmean(tissue_PQ_bin))

    # Save per-tissue metrics as a csv file
    for_dataframe = {'Tissue name': tissue_types + ['mean'],
                        'PQ': all_tissue_mPQ + [np.nanmean(all_tissue_mPQ)] , 'PQ bin': all_tissue_bPQ + [np.nanmean(all_tissue_bPQ)]}
    df = pd.DataFrame(for_dataframe, columns=['Tissue name', 'PQ', 'PQ bin'])
    df.to_csv(save_path + '/tissue_stats.csv')

    # Show overall metrics - mPQ is average PQ over the classes and the tissues, bPQ is average binary PQ over the tissues
    print('-' * 40)
    print('Average mPQ:{}'.format(np.nanmean(all_tissue_mPQ)))
    print('Average bPQ:{}'.format(np.nanmean(all_tissue_bPQ)))

#####
if __name__ == '__main__':
    # args = docopt.docopt(__doc__, version='PanNuke Evaluation v1.0')
    main()

