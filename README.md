# HA2P-Net


## 1. Installation

Clone this repo.

This code requires PyTorch 1.10+ and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```


## 2. Data preparation

For small dataset Kumar and CoNSeP, can be found [here](https://drive.google.com/file/d/1_eI_ii6xcNe_77NWx7Qo8_KndK5UwPBO/view?usp=sharing)

The [PanNuKe](https://arxiv.org/pdf/2003.10778v7.pdf) datasets can be found [here](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke)

Download and unzip all the files where the folder structure should look this:

```none
HA2P-Net
├── ...
├── datasets
│   ├── kumar
│   │   ├── train
│   │   ├── test
│   ├── CoNSeP
│   │   ├── train
│   │   ├── test
│   ├── PanNuKe
│   │   ├── images
│   │   │   ├── fold1
│   │   │   │   ├── images.npy
│   │   │   │   ├── types.npy
│   │   │   ├── fold2
│   │   │   │   ├── images.npy
│   │   │   │   ├── types.npy
│   │   │   ├── fold3
│   │   │   │   ├── images.npy
│   │   │   │   ├── types.npy
│   │   ├── masks
│   │   │   ├── fold1
│   │   │   │   ├── masks.npy
│   │   │   ├── fold2
│   │   │   │   ├── masks.npy
│   │   │   ├── fold3
│   │   │   │   ├── masks.npy
├── ...
```

## 3. Training and Inference
To reproduce the performance, you need one 3090 GPU at least.

Download ImageNet Pretrain Checkpoints from [here](https://drive.google.com/file/d/1PKC7k4Ls_ASabSQOoHNQR-V-VvAAjMVX/view?usp=drive_link)

<details>
  <summary>
    <b>1) Kumar Dataset</b>
  </summary>

run the command to train the model
```bash
python train.py --name=kumar_exp --seed=888 --config=configs/kumar_notype_effv2s.yaml
```

run the command to inference
```bash
python inference.py --name=kumar_exp
```
</details>

<details>
  <summary>
    <b>2) CoNSeP Dataset</b>
  </summary>

run the command to train the model
```bash
python train.py --name=consep_exp --seed=888 --config=configs/consep_type_effv2s.yaml
```

run the command to inference
```bash
python inference.py --name=consep_exp
```
</details>


<details>
  <summary>
    <b>2)  PanNuKe Dataset</b>
  </summary>

run the command to train the model
```bash
python train_pannuke.py --name=pannuke_exp --seed=888 --train_fold={} --val_fold={} --test_fold={} --config=configs/effv2s.yaml
```
[train_fold, val_fold, test_fold] should be selected from {[1, 2, 3], [2, 1, 3], [3, 2, 1]}

run the command to inference the model
```bash
python infer_pannuke.py --name=pannuke_exp --train_fold={} --test_fold={}
```

run the command to evaluate the performance
```bash
python eval_pannuke.py --name=pannuke_exp --train_fold={} --test_fold={}
```

</details>


## Citation
If our work or code helps you, please consider to cite our paper. Thank you!

