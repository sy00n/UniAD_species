### 수정된 부분 설명
- Official repository에서 구현된 datasets/custom_dataset.py 파일은 사실상 mvtec 구조로 저장된 데이터를 위한 loader임.
- custom_dataset.py로 데이터를 로딩하려먼 meta files(json파일들)을 만들어서 불러오는 식으로 구성되어 있는데, ood detection의 경우 test anomaly를 불러오는 부분에 있어서 json 파일을 모두 읽어오는 방식으로 수행 시, 너무 비효율적, evaluation 시 시간이 심각하게 소요됨)
- 데이터 구조를 mvtec 구조로 강제화해야지만 실행 가능하다는 점도 비효율적임
- 따라서 cifar10, species, imagenet과 같은 **natural image loader를 추가로 구현함**
  - cifar dataloader(datasets/cifar_dataset.py) 같은 경우도 마찬가지로 meta 파일들을 활용하는 방식으로 load되게 구현되어 있고 torchvision으로부터 데이터를 받아오게 되어있기 때문에 natural image를 위한 loader 구현이 필요한 상황이었음.
  - 하지만 여전히 아쉬운 점은, species와 imagenet의 경우 데이터가 저장된 구조가 다르기 때문에 이 역시 각각 개별적인 dataloader로부터 path를 읽어오게끔 구현함. 따라서 이런 한계를 해결하려면 pytorch에서 제공하는 imagefolder를 사용하면 될 것 같지만, 그부분은 우선 보류함. 

# UniAD
Official PyTorch Implementation of [A Unified Model for Multi-class Anomaly Detection](https://arxiv.org/abs/2206.03687), Accepted by NeurIPS 2022 Spotlight.

![Image text](docs/setting.jpg)
![Image text](docs/res_mvtec.jpg)

## 1. Quick Start

### 1.1 MVTec-AD

- **Create the MVTec-AD dataset directory**. Download the MVTec-AD dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad). Unzip the file and move some to `./data/MVTec-AD/`. The MVTec-AD dataset directory should be as follows. 

```
|-- data
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
        |-- json_vis_decoder
        |-- train.json
        |-- test.json
```

- **cd the experiment directory** by running `cd ./experiments/MVTec-AD/`. 

- **Train or eval** by running: 

    (1) For slurm group:  `sh train.sh #NUM_GPUS #PARTITION` or `sh eval.sh #NUM_GPUS #PARTITION`.

    (2) For torch.distributed.launch:  `sh train_torch.sh #NUM_GPUS #GPU_IDS` or `sh eval_torch.sh #NUM_GPUS #GPU_IDS`, *e.g.*, train with GPUs 1,3,4,6 (4 GPUs in total): `sh train_torch.sh 4 1,3,4,6`.

    **Note**: During eval, please *set config.saver.load_path* to load the checkpoints. 

- **Results and checkpoints**. 

| Platform | GPU | Detection AUROC | Localization AUROC | Checkpoints | Note |
| ------ | ------ | ------ | ------ | ------ | ------ | 
| slurm group | 8 GPUs (NVIDIA Tesla V100 16GB)|  96.7 | 96.8 | [here](https://drive.google.com/file/d/1q03ysv_5VJATlDN-A-c9zvcTuyEeaQHG/view?usp=sharing) | ***A unified model for all categories*** |
| torch.distributed.launch | 1 GPU (NVIDIA GeForce GTX 1080 Ti 11 GB)|  97.6 | 97.0 | [here](https://drive.google.com/file/d/1v282ZlibC-b0H9sjLUlOSCFNzEv-TIuh/view?usp=sharing) | ***A unified model for all categories*** |


### 1.2 CIFAR-10

- **Create the CIFAR-10 dataset directory**. Download the CIFAR-10 dataset from [here](http://www.cs.toronto.edu/~kriz/cifar.html). Unzip the file and move some to `./data/CIFAR-10/`. The CIFAR-10 dataset directory should be as follows. 

```
|-- data
    |-- CIFAR-10
        |-- cifar-10-batches-py
```

- **cd the experiment directory** by running `cd ./experiments/CIFAR-10/01234/`. Here we take class 0,1,2,3,4 as normal samples, and other settings are similar.

- **Train or eval** by running: 

    (1) For slurm group:  `sh train.sh #NUM_GPUS #PARTITION` or `sh eval.sh #NUM_GPUS #PARTITION`.

    (2) For torch.distributed.launch:  `sh train_torch.sh #NUM_GPUS #GPU_IDS` or `sh eval_torch.sh #NUM_GPUS #GPU_IDS`.

    **Note**: During eval, please *set config.saver.load_path* to load the checkpoints. 

- **Results and checkpoints**. Training on 8 GPUs (NVIDIA Tesla V100 16GB) results in following performance.

| Normal Samples | {01234} | {56789} | {02468} | {13579} | Mean |
| ------ | ------ | ------ | ------ | ------ | ------ |
| AUROC | 84.4 | 79.6 | 93.0 | 89.1 | 86.5 |


## 2. Visualize Reconstructed Features

We **highly recommend** to visualize reconstructed features, since this could directly prove that our UniAD *reconstructs anomalies to their corresponding normal samples*. 

### 2.1 Train Decoders for Visualization

- **cd the experiment directory** by running `cd ./experiments/train_vis_decoder/`. 

- **Train** by running: 

    (1) For slurm group:  `sh train.sh #NUM_GPUS #PARTITION`.

    (2) For torch.distributed.launch: `sh train_torch.sh #NUM_GPUS #GPU_IDS #CLASS_NAME`.

    **Note**: for torch.distributed.launch, you should *train one vis_decoder for a specific class for one time*. 

### 2.2 Visualize Reconstructed Features

- **cd the experiment directory** by running `cd ./experiments/vis_recon/`. 

- **Visualize** by running (only support 1 GPU): 

    (1) For slurm group:  `sh vis_recon.sh #PARTITION`.

    (2) For torch.distributed.launch:  `sh vis_recon_torch.sh #CLASS_NAME`.

    **Note**: for torch.distributed.launch, you should *visualize a specific class for one time*. 

## 3. Questions

### 3.1 Explanation of Evaluation Results

The first line of the evaluation results are shown as follows. 

|  clsname   |   pixel  |   mean   |   max    |   std    |
|:----------:|:--------:|:--------:|:--------:|:--------:|

The *pixel* means anomaly localization results. 

The *mean*, *max*, and *std* mean **post-processing methods** for anomaly detection. That is to say, the anomaly localization result is an anomaly map with the shape of *H x W*. We need to *convert this map to a scalar* as the anomaly score for this whole image. For this convert, you have 3 options: 

- use the *mean* value of the anomaly map.
- use the *max* value of the (averagely pooled) anomaly map.
- use the *std* value of the anomaly map.

In our paper, we use *max* for MVTec-AD and *mean* for CIFAR-10. 

### 3.2 Visualize Learned Query Embedding

If you have finished the training of the main model and decoders (used for visualization) for MVTec-AD, you could also choose to visualize the learned query embedding in the main model. 

- **cd the experiment directory** by running `cd ./experiments/vis_query/`. 

- **Visualize** by running (only support 1 GPU): 

    (1) For slurm group:  `sh vis_query.sh #PARTITION`.

    (2) For torch.distributed.launch:  `sh vis_query_torch.sh #CLASS_NAME`.

    **Note**: for torch.distributed.launch, you should *visualize a specific class for one time*. 

Some results are very interesting. The learned query embedding partly contains some features of normal samples. However, we ***did not*** fully figure out this and this part ***was not*** included in our paper. 

![Image text](docs/query_bottle.jpg)
![Image text](docs/query_capsule.jpg)

## Acknowledgement

We use some codes from repositories including [detr](https://github.com/facebookresearch/detr) and [efficientnet](https://github.com/lukemelas/EfficientNet-PyTorch). 
