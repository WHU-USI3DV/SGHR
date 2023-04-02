## Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting

[Haiping Wang](https://hpwang-whu.github.io/), [Yuan Liu](https://liuyuan-pal.github.io/), [Zhen Dong](https://dongzhenwhu.github.io/), [Yulan Guo](http://yulanguo.me/), [Yu-Shen Liu](https://yushen-liu.github.io/), [Wenping Wang](https://www.cs.hku.hk/people/academic-staff/wenping), [Bisheng Yang](http://3s.whu.edu.cn/info/1025/1415.htm)

In this paper, we present a new method for the multiview registration of point cloud. Previous multiview registration methods rely on exhaustive pairwise registration to construct a densely-connected pose graph and apply Iteratively Reweighted Least Square (IRLS) on the pose graph to compute the scan poses. However, constructing a densely-connected graph is time-consuming and contains lots of outlier edges, which makes the subsequent IRLS struggle to find correct poses. To address the above problems, we first propose to use a neural network to estimate the overlap between scan pairs, which enables us to construct a sparse but reliable pose graph. Then, we design a novel history reweighting function in the IRLS scheme, which has strong robustness to outlier edges on the graph. In comparison with existing multiview registration methods, our method achieves $11$\% higher registration recall on the 3DMatch dataset and $\sim13$\% lower registration errors on the ScanNet dataset while reducing $\sim70$\% required pairwise registrations. Comprehensive ablation studies are conducted to demonstrate the effectiveness of our designs.


- [Preprint paper](). Coming soon!
<!-- - [Project page](https://whu-usi3dv.github.io/SGHR/). -->


## News
- 2023-04-01: The code of SGHR is released.
- 2023-02-28: SGHR is accepted by CVPR 2023! :tada: :tada:


## Requirements
Here we offer the [YOHO](https://github.com/HpWang-whu/YOHO) backbone SGHR. Thus YOHO requirements need to be met:
- Ubuntu 14.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.6 or higher

Specifically, The code has been tested with:
- Ubuntu 16.04, CUDA 11.1, python 3.7.10, Pytorch 1.7.1, GeForce RTX 2080Ti.

## Installation
- First, create the conda environment:
  ```
  conda create -n sghr python=3.7
  conda activate sghr
  ```

- Second, intall Pytorch. We have checked version 1.7.1 and other versions can be referred to [Official Set](https://pytorch.org/get-started/previous-versions/).
  ```
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
  ```

- Third, install other packages, here we use 0.8.0.0 version [Open3d](http://www.open3d.org/) for Ubuntu 16.04:
  ```
  pip install -r requirements.txt
  ```

- Finally, compile the [CUDA based KNN searcher](https://github.com/vincentfpgarcia/kNN-CUDA):
  ```
  cd knn_search/
  export CUDA_HOME=/usr/local/cuda-11.1 #We have checked cuda-11.1.
  python setup.py build_ext --inplace
  cd ..
  ```

## Dataset & Pretrained model
The datasets are accessible in [BaiduDesk](https://pan.baidu.com/s/1FcAPjmrsJ6EEPLbtf85Irw)(Code:oouk) and Google Cloud:

Trainset: 
- [3DMatch_train](https://drive.google.com/file/d/1ObVWsvZ0IyjWRBaCdQb_1BZOmKINht5A/view?usp=sharing);

Testset:
- [3DMatch/3DLomatch](https://drive.google.com/file/d/1T9fyU2XAYmXwiWZif--j5gP9G8As5cxn/view?usp=sharing);
- [ScanNet](https://drive.google.com/file/d/1GM6ePDDqZ3awJOZpctd3nqy1VgazV6CD/view?usp=sharing);
- [ETH](https://drive.google.com/file/d/1MW8SV44fuFTS5b2XrdADaqH5xRf3sLMk/view?usp=sharing).

Datasets above contain the point clouds (.ply), keypoints (.txt, 5000 per point cloud), and rotation-invariant yoho-desc(.npy, extracted on the keypoints) files. Please place the data to ```./data``` following the example data structure as:

```
data/
├── 3dmatch/
    └── kitchen/
        ├── PointCloud/
            ├── cloud_bin_0.ply
            ├── gt.log
            └── gt.info
        ├── yoho_desc/
            └── 0.npy
        └── Keypoints/
            └── cloud_bin_0Keypoints.txt
├── 3dmatch_train/
├── scannet/
└── ETH/
```

## Train
You can train SGHR with the 3dmatch_train dataset downloaded above, where we offer the 32-dim **rotation-invariant** yoho-desc we extracted on 3dmatch_train and you can also extract 32-dim [invariant yoho-desc](https://github.com/HpWang-whu/YOHO)(row-pooling on yoho-desc) yourself and save the features to '''data/3dmatch_train/\<scene\>/yoho_desc'''.
Then, you can train SGHR with the following commond:
```
python Train.py
```

## Test
To evalute SGHR on 3DMatch and 3DLoMatch, you can use the following commands:
```
# extract global features
python Test.py --dataset 3dmatch
# conduct multiview registration
python Test_cycle.py --dataset 3dmatch --rr
```

To evalute SGHR on ScanNet, you can use the following commands:
```
python Test.py --dataset scannet
python Test_cycle.py --dataset scannet --ecdf
```

To evalute SGHR on ETH, you can use the following commands:
```
python Test.py --dataset ETH
python Test_cycle.py --dataset ETH --topk 6 --inlierd 0.2 --tau_2 0.5 --rr
```

To evalute SGHR on your own dataset, you can follow [here](data/Readme.md).

## Citation

Please consider citing SGHR if this program benefits your project
```

```

## Other Projects
Welcome to take a look at the homepage of our research group [WHU-USI3DV](https://github.com/WHU-USI3DV) !
