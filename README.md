# Light Field Super-Resolution Using Decoupled Selective Matching

**Official implementation** of the following paper

Yutong Liu, Zhen Cheng, Zeyu Xiao, and Zhiwei Xiong, Light Field Super-Resolution Using Decoupled Selective Matching

<!-- [Light Field Super-Resolution with Zero-Shot Learning](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Light_Field_Super-Resolution_With_Zero-Shot_Learning_CVPR_2021_paper.html). In CVPR 2021. (Oral) -->
<!-- [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Light_Field_Super-Resolution_With_Zero-Shot_Learning_CVPR_2021_paper.pdf) | [Bibtex](https://github.com/Joechann0831/LFZSSR#citation) -->

[Paper]() | [Bibtex]()



## Dependencies

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- Pytorch 1.7.0
- einops
- Numpy
- Scipy
- matplotlib
- TensorboardX
- MATLAB (For data preparation)



## Usage

### 1. Dataset Preparation

- We make experiments on two benchmarks CiytU and BasicLFSR.

#### 1.1 CiytU

- For the benchmark CiytU, please refer to [ATO](https://github.com/jingjin25/LFSSR-ATO) or [SAV_conv](https://github.com/Joechann0831/SAV_conv) for the preparetion of the dataset. You can downland the test dataset from [BaiduYun](https://pan.baidu.com/s/13W_r0Bk68TUXwSflWch01A?pwd=ustc) and put them into the folder *./CiytU/data/* for a readily start. 

#### 1.2 BasicLFSR

- For the benchmark BasicLFSR, please refer to [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR) for the preparetion of the dataset. You can downland the test dataset from [BaiduYun](https://pan.baidu.com/s/1Ip5L-mFFg7vAK3IK8Se8QA?pwd=ustc) and put them into the folder *./BasicLFSR/data/* for a readily start. 


### 2. Pretrained Model Preparation
- For the Pretrained Model, please downland checkpoint from [BaiduYun](https://pan.baidu.com/s/16-jSWglhZxVLnLOQXIugWA?pwd=ustc) and put them into the folder *./CiytU/pretrained_model/*, while please downland checkpoint from [BaiduYun](https://pan.baidu.com/s/1kfqLSIyPY41PcCVSnpT5yg?pwd=ustc) and put them into the folder *./BasicLFSR/pretrained_model/*.
### 3. Train & test

For CityU, to train and test our DSMNet under the scale of 2 as an example:
```shell
cd ./CityU/
bash train_CityU_scale2.sh
bash test_CityU_scale2.sh
```
For BasicLFSR, to train and test our DSMNet under the scale of 2 as an example:
```shell
cd ./BasicLFSR/
bash train_BasicLFSR_scale2.sh
bash test_BasicLFSR_scale2.sh
```

## Citation

If you find this work helpful, please consider citing our paper.
```latex
@ARTICLE{10268449,
  author={Liu, Yutong and Cheng, Zhen and Xiao, Zeyu and Xiong, Zhiwei},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Light Field Super-Resolution Using Decoupled Selective Matching}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3321085}}
```

## Related Projects
[ATO](https://github.com/jingjin25/LFSSR-ATO)

[SAV_conv](https://github.com/Joechann0831/SAV_conv)

[BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)


## Contact

If you have any problem about the released code, please contact me with email (ustclyt@mail.ustc.edu.cn).
