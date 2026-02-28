# BiNAR: A Bi-Modal Framework for Non-Aligned RGB-IR 3D Reconstruction via Gaussian Splatting
![loading gif](./pics/desktop.gif)

BiNAR achieves pixel-level aligned RGB-IR bi-modal 3D scene reconstruction and rendering.

## Setup
Clone this repository and set up the environment with the following command:
```
git clone git@github.com:jankin-wang/BiNAR.git
cd BiNAR

conda create -y -n binar python=3.8
conda activate binar

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
```

## Dataset
### PARID_Raw for Training
Please download the raw data from [PARID_Raw](https://huggingface.co/datasets/jankinkin/BiNAR_data/tree/main/PARID_Raw) and place it in the `./dataset/PARID_Raw` folder under the project directory.

### PARID Dataset
The [PARID (Pixel-Aligned RGB-IR Dataset)](https://huggingface.co/datasets/jankinkin/BiNAR_data/tree/main/PARID) provides pixel-level aligned RGB and IR image pairs across both indoor and outdoor scenes. Each IR image retains real thermal information. If you need to recover the temperature information of each pixel in the scene, use the temperature range in the table below to perform inverse normalization.


|   Scene  |   Type  | Temperature Min (°C) | Temperature Max (°C) |
| :------: | :-----: | :------------------: | :------------------: |
|  Desktop |  Indoor |         0         |         80         |
|    UAV   | Indoor |         14         |         34         |
|  Kettles |  Indoor |         7         |         33         |
| Computer |  Indoor |         10         |         60         |
|  Aircon  |  Indoor |         1         |         50         |
|  Apples  |  Indoor |         -5         |         30         |
|  Bottles |  Indoor |         -6         |         30         |
|  E-Bike  | Outdoor |         5         |         24         |
|    Car   | Outdoor |         5         |         21         |
|  Bicycle | Outdoor |         5         |         25         |


## Quick Start
To start training, rendering and evaluating, simply use:

`python scripts/run_joint.py`

## Citation
If you find our work useful in your research, please consider citing:
、、、
@inproceedings{wang2026binar,
  title={BiNAR: A Bi-Modal Framework for Non-Aligned RGB-IR 3D Reconstruction via Gaussian Splatting},
  author={Wang, Zhongwen and Ling, Han and Zhang, Weihao and Sun, Yinghui and Sun, Quansen},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4407--4416},
  year={2026}
}
、、、
