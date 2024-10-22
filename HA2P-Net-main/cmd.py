import subprocess

# 定义不同的 fold 值列表
train_folds = [2,1,3]
val_folds = [1,2,2]
test_folds = [3,3,1]

for train_fold, val_fold, test_fold in zip(train_folds, val_folds, test_folds):
    cmd = f"python train_pannuke.py --name='pannuke_exp_mobilenetL_my' --logdir='logs/pre_mobilenetL_my' --seed=888 --train_fold={train_fold} --val_fold={val_fold} --test_fold={test_fold}"
    subprocess.call(cmd, shell=True)
##pannuke_exp_effv2s_my_noife/nohaff