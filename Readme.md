<h1 align="center"> <p>😍 SGHR</p></h1>
<h3 align="center">
<a href="https://arxiv.org/abs/2304.00467" target="_blank">Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting</a>
</h3>

<h3 align="center">
CVPR 2023
</h3>

<p align="center">
<a href="https://hpwang-whu.github.io/" target="_blank">Haiping Wang</a><sup>*,1</sup>, 
<a href="https://liuyuan-pal.github.io/" target="_blank">Yuan Liu</a><sup>*,2</sup>,
<a href="https://dongzhenwhu.github.io/" target="_blank">Zhen Dong</a><sup>&dagger;,1</sup>, 
<a href="http://yulanguo.me/" target="_blank">Yulan Guo</a><sup>3</sup>,
<a href="https://yushen-liu.github.io/" target="_blank">Yu-Shen Liu</a><sup>4</sup>,
<a href="https://www.cs.hku.hk/people/academic-staff/wenping" target="_blank">Wenping Wang</a><sup>5</sup>
<a href="https://3s.whu.edu.cn/info/1025/1415.htm" target="_blank">Bisheng Yang</a><sup>&dagger;,1</sup> <br>
</p>

<p align="center">
<sup>1</sup>Wuhan University &nbsp;&nbsp; 
<sup>2</sup>The University of Hong Kong &nbsp;&nbsp; 
<sup>3</sup>Sun Yat-sen University &nbsp;&nbsp; <br>
<sup>4</sup>Tsinghua University &nbsp;&nbsp; 
<sup>5</sup>Texas A&M University &nbsp;&nbsp; <br>
<sup>*</sup>The first two authors contribute equally. &nbsp;&nbsp; 
<sup>&dagger;</sup>Corresponding authors. &nbsp;&nbsp; 
</p>

In this paper, we present a new method for the multiview registration of point cloud. Previous multiview registration methods rely on exhaustive pairwise registration to construct a densely-connected pose graph and apply Iteratively Reweighted Least Square (IRLS) on the pose graph to compute the scan poses. However, constructing a densely-connected graph is time-consuming and contains lots of outlier edges, which makes the subsequent IRLS struggle to find correct poses. To address the above problems, we first propose to use a neural network to estimate the overlap between scan pairs, which enables us to construct a sparse but reliable pose graph. Then, we design a novel history reweighting function in the IRLS scheme, which has strong robustness to outlier edges on the graph. In comparison with existing multiview registration methods, our method achieves $11$\% higher registration recall on the 3DMatch dataset and $\sim13$\% lower registration errors on the ScanNet dataset while reducing $\sim70$\% required pairwise registrations. Comprehensive ablation studies are conducted to demonstrate the effectiveness of our designs.

<p align="center">
 | 
<a href="https://arxiv.org/abs/2304.00467" target="_blank">Paper</a> | 
<a href="./utils/media/sghr_poster.png" target="_blank">Poster</a> | 
<a href="https://www.youtube.com/watch?v=TGoCD4QqKEg" target="_blank">Video</a>
 | 
</p>

## 🆕 News
- 2023-05-13: An introduction video of SGHR on YouTube.
- 2023-04-04: Release SGHR on Arxiv. 
- 2023-04-01: The code of SGHR is released.
- 2023-02-28: SGHR is accepted by CVPR 2023! 🎉🎉

## ✨ Pipeline

<img src="utils/media/pipeline.png" alt="Network" style="zoom:50%;">

## 💻 Requirements
Here we offer the [YOHO](https://github.com/HpWang-whu/YOHO) backbone SGHR. Thus YOHO requirements need to be met:
- Ubuntu 14.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.6 or higher

Specifically, The code has been tested with:
- Ubuntu 16.04, CUDA 11.1, python 3.7.10, Pytorch 1.7.1, GeForce RTX 2080Ti.
- Ubuntu 20.04, CUDA 11.1, python 3.7.16, Pytorch 1.10.0, GeForce RTX 4090.

## 🔧 Installation
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

## 💾 Dataset & Pretrained model
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

## 🚅 Train
You can train SGHR with the 3dmatch_train dataset downloaded above, where we offer the 32-dim **rotation-invariant** yoho-desc we extracted on 3dmatch_train and you can also extract 32-dim [invariant yoho-desc](https://github.com/HpWang-whu/YOHO)(row-pooling on yoho-desc) yourself and save the features to '''data/3dmatch_train/\<scene\>/yoho_desc'''.
Then, you can train SGHR with the following commond:
```
python Train.py
```

## ✏️ Test
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

## 💡 Citation

Please consider citing SGHR if this program benefits your project
```
@inproceedings{
wang2023robust,
title={Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting},
author={Haiping Wang and Yuan Liu and Zhen Dong and Yulan Guo and Yu-Shen Liu and Wenping Wang and Bisheng Yang},
booktitle={Conference on Computer Vision and Pattern Recognition},
year={2023}
}
```

## 🔗 Related Projects
Take a look at our previous works on [feature extraction](https://github.com/HpWang-whu/YOHO) and [pairwise registration](https://github.com/HpWang-whu/RoReg)!