# P-Net for Anomaly Detection (Pytorch)

This is the implementation of the paper:

Kang Zhou, Yuting Xiao, Jianlong Yang, Jun Cheng, Wen Liu, Weixin Luo, Zaiwang Gu, Jiang Liu, Shenghua Gao. Encoding Structure-Texture Relation with P-Net for Anomaly Detection in Retinal Images. ECCV 2020.

using PyTorch.

**If you have any question, please feel free to contact us ({zhoukang, xiaoyt}@shanghaitech.edu.cn).**


## Introduction

![avatar](figures/intro.png)

The motivation of leveraging structure information for anomaly detection. The normal medical images are highly structured, while the regular structure is broken in abnormal images. For example, the lesions (denoted by black bounding box and red arrow in (a) of diabetic retinopathy destroy the blood vessel and histology layer in retina. Thus, in the abnormal retinal fundus image and optical coherence tomography (OCT) image, the lesions (denoted by red color in (b) and (c)) broke the structure. Moreover, this phenomenon agrees with the cognition of doctors. Motivated by this clinical observation, we suggest utilizing the structure information in anomaly detection. 

## Method

![avatar](figures/method.png)

The pipeline of our P-Net.

![avatar](figures/method_da.png) 

(a) Structure extraction network with domain adaptation (DA). (b) The qualitative results of DA for OCT images. The structure of target image cannot be extracted well without DA. 

![avatar](figures/mvtech.png)      
                           
Qualitative results of the images in MV-Tech AD dataset.


## Getting started

### Environment
Python 3.5.2  
Pytorch 1.1.0  
torchvision 0.2.1

### To Do
- [ ] Update the P_Net_v1.

- [ ] Update the dataloader of RESC dataset. 

- [ ] Update the MVTec AD dataset.

<!--### Getting the datasets-->

<!--The PF-Pascal dataset (used for training and evaluation) can be downloaded and unzipped by browsing to the `datasets/pf-pascal/` folder and running `download.sh`.-->

<!--The PF-Willow and TSS dataset (used for evaluation) can be downloaded by browsing to the `datasets/` folder and run `download_datasets.py`. The datasets will be under `datasets/proposal-flow-willow` and `datasets/tss`-->



<!--### Getting the trained models-->

<!--The trained models trained on PF-Pascal (`best_dccnet.pth.tar`) can be dowloaded by browsing to the `trained_models/` folder and running `download.sh` (comming soon).-->


<!--## Training-->

<!--To train a model, run `train_dccnet.sh` under `scripts` folder to reproduce our results.-->


<!--## Evaluation-->

<!--Evaluation for PF-Pascal and PF-Willow is implemented in the `eval_pf_pascal.py` and `eval_pf_willow.py` file respectively. You can run the evaluation in the following way: -->

<!--```bash-->
<!--python eval_pf_pascal.py --checkpoint trained_models/[checkpoint name]-->
<!--```-->

<!--Evaluation for TSS is implemented in the `eval_tss.py` file. You can run the evaluation in the following way: -->

<!--```bash-->
<!--python eval_tss.py --checkpoint trained_models/[checkpoint name]-->
<!--```-->

<!--This will generate a series of flow files in the `datasets/dccnet_results` folder that then need to be fed to the TSS evaluation Matlab code. -->
<!--In order to run the Matlab evaluation, you need to clone the [TSS repo](https://github.com/t-taniai/TSS_CVPR2016_EvaluationKit) and follow the corresponding instructions.-->

<!--## Acknwoledgement-->

<!--We borrow tons of code from [NC-Net](https://github.com/ignacio-rocco/ncnet) and [WeakAlign](https://github.com/ignacio-rocco/weakalign).-->

## BibTeX 


If you use this code in your project, please cite our paper:
````
@inproceedings{zhou2020encoding,
  title={Encoding Structure-Texture Relation with P-Net for Anomaly Detection in Retinal Images},
  author={Zhou, Kang and Xiao, Yuting and Yang, Jianlong and Cheng, Jun and Liu, Wen and Luo, Weixin and Gu, Zaiwang and Liu, Jiang and Gao, Shenghua.},
  booktitle={ECCV},
  year={2020}
}
````


