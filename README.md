# Trexplorer: Recurrent DETR for Topologically Correct Tree Centerline Tracking

This repository provides the official implementation of the [Trexplorer: Recurrent DETR for Topologically Correct Tree Centerline Tracking](https://github.com/RomStriker/Trexplorer) paper by [Roman Naeem](https://research.chalmers.se/en/person/nroman), [David Hagerman](https://research.chalmers.se/en/person/olzond), [Lennart Svensson](https://research.chalmers.se/person/pale) and [Fredrik Kahl](https://www.chalmers.se/personer/kahlf/). The codebase builds upon [DETR](https://github.com/facebookresearch/detr) and [Trackformer](https://github.com/timmeinhardt/trackformer).

<!-- **As the paper is still under submission this repository will continuously be updated and might at times not reflect the current state of the [arXiv paper](https://arxiv.org/abs/2012.01866).** -->

<div align="center">
    <img src="docs/architecture.png" alt="arch" width="1000"/>
</div>

## Abstract

Tubular structures with tree topology such as blood vessels, lung airways, and more are abundant in human anatomy. Tracking these structures with correct topology is crucial for many downstream tasks that help in early detection of conditions such as vascular and pulmonary diseases. Current methods for centerline tracking suffer from predicting topologically incorrect centerlines and complex model pipelines. To mitigate these issues we propose Trexplorer, a recurrent DETR based model that tracks topologically correct centerlines of tubular tree objects in 3D volumes using a simple model pipeline. We demonstrate the model's performance on a publicly available synthetic vessel centerline dataset and show that our model outperforms the state-of-the-art on centerline topology and graph-related metrics, and performs well on detection metrics.

## Installation
1. Clone this repository:
    ```
    git clone https://github.com/RomStriker/Trexplorer.git
    ``` 
2. Install requirements:
    ```
    pip install -r requirements.txt
    ```
3. Install PyTorch 2.2 with CUDA 11.8:
    ```
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
    ```

## Training

### Data Preparation
1. Download the [Synthetic Vessel Dataset](https://github.com/giesekow/deepvesselnet/wiki/Datasets/).
2. Convert the graph vtk files to the required annotation format.
3. Create the following directory structure:
    ```
    Trexplorer
    ├── data
    │   ├── synthetic
    │   │   ├── annots_train
    │   │   ├── annots_val
    │   │   ├── annots_val_sub_vol
    │   │   ├── annots_test 
    │   │   ├── images_train
    │   │   ├── images_val
    │   │   ├── images_val_sub_vol
    │   │   ├── images_test
    │   │   ├── masks_train
    │   │   ├── masks_val
    │   │   ├── masks_val_sub_vol
    │   │   ├── masks_test
    │   │   ├── annots_val_sub_vol.pickle
    ```
   The '_train', '_val', and '_test' directories contain the training, validation, and test images respectively. The '_val_sub_vol' directory contains the validation images that are used for patch-level evaluation. The 'masks' directories contain the binary masks of the vessel trees. The 'annots' directories contain the annotation files in the required format. The 'annots_val_sub_vol.pickle' file contains the annotations for the validation sub-volume images. 

### Training
The training script uses the configuration file `./configs/train.yaml` to set the hyperparameters. To train the model, run the following command from the root directory:
```
python ./src/train_trx.py
```
For distributed training, use the following command:
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS ./src/train_trx.py
```

## Evaluation
For evaluation, in addition to `./configs/train.yaml`, we use  `./configs/eval.yaml`. To evaluate the model, run the following command from the root directory:
```
python ./src/train_trx.py with eval
```
For distributed training, use the following command:
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS ./src/train_trx.py with eval
```

## Publication
If you use this software in your research, please cite our publication:

```
@InProceedings{10.1007/978-3-031-72120-5_69,
author="Naeem, Roman
and Hagerman, David
and Svensson, Lennart
and Kahl, Fredrik",
title="Trexplorer: Recurrent DETR for Topologically Correct Tree Centerline Tracking",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="744--754",
abstract="Tubular structures with tree topology such as blood vessels, lung airways, and more are abundant in human anatomy. Tracking these structures with correct topology is crucial for many downstream tasks that help in early detection of conditions such as vascular and pulmonary diseases. Current methods for centerline tracking suffer from predicting topologically incorrect centerlines and complex model pipelines. To mitigate these issues we propose Trexplorer, a recurrent DETR based model that tracks topologically correct centerlines of tubular tree objects in 3D volumes using a simple model pipeline. We demonstrate the model's performance on a publicly available synthetic vessel centerline dataset and show that our model outperforms the state-of-the-art on centerline topology and graph-related metrics, and performs well on detection metrics. The code is available at https://github.com/RomStriker/Trexplorer.",
isbn="978-3-031-72120-5"
}
```