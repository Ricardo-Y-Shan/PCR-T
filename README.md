# PCR-T

A PyTorch implementation of the article 'A Coarse-to-Fine Transformer-Based Network for 3D Reconstruction from Non-Overlapping Multi-View Images', which has been published on Remote Sensing, 2024, 16, 901. [Remote Sensing | Free Full-Text | A Coarse-to-Fine Transformer-Based Network for 3D Reconstruction from Non-Overlapping Multi-View Images (mdpi.com)](https://www.mdpi.com/2072-4292/16/5/901)



## Requirements

- python=3.7
- numpy
- torch=1.12.1+cu113
- torchvision=0.13.1+cu113
- opencv-python

See more in requirements.txt.



## CUDA Extensions

Before training and testing, please compile CUDA ops for Chamfer Distance and Earth Mover Distance.

- In `cuda/chamfer/`

```
!pip install .
!python test.py
```

- In `cuda/emd/`

```
!pip install .
!python test.py
```

These two ops are packaged in `model/layer/`.

The knn op is implemented by pytorch in `model/layer/knn_wrapper.py`.



## Dataset

- We use ShapeNet as our training and testing data, following [GitHub - walsvid/Pixel2MeshPlusPlus: Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation. In ICCV2019.](https://github.com/walsvid/Pixel2MeshPlusPlus/tree/master). Please download the rendering images and G.T. point clouds following the dataset guideline of this repo, and then modify their path in the config file `global_config.py`.
- On the basis of this dataset, we resample the G.T. point clouds in the same number (3072 points) for testing. Please modify the "resample_gt_dir" of "SHAPENET_PATH" in `global_config.py` and then run the script `utils/farthest_point_sample.py`.
- We also use Pix3D for robustness testing. Please download the dataset following the repo link [GitHub - xingyuansun/pix3d: Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling](https://github.com/xingyuansun/pix3d), and then modify the path in the config file `global_config.py`.



## Run

Our model consists of two parts: a PSGN module and a Transformer module. These two modules should be trained separately. Please modify the different config file `options_psgn.py` and `options_pcr.py` and then turn to following main entrance.

- Train: `main_train.py`
- Test: `main_test.py`
- Predict: `main_predict.py`