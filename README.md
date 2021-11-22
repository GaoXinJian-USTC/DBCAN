# DBCAN: Dual-Branch Cross-attention Newtwork for Scene Text Recognition

> [Paper Link](#)


## Requirements
```
conda create -n DBCAN python=3.8.10

conda install pytorch=1.8.0 cudatoolkit=11.0 torchvision=0.9.0

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.8.0/index.html

pip install mmdet

git clone git@github.com:Gaoxj2020/DBCAN.git

cd DBCAN && pip install -r requirements.txt
```
## Data Preparing

- For Training data and Test data, just follow the instructions in https://mmocr.readthedocs.io/en/latest/datasets/recog.html

- Specially,for IC15 and IC13, We use the protocol proposed in [1], we provide the two datasets on 

- > [BaiduYun](https://pan.baidu.com/s/1eUjlnX7wf1sQG8NYGaMNzA) key:u2jh


## For Test

We provide code for using our pretrained model to recognize text images.

- Our model can be downloaded via Baidu net disk: [download_link](https://pan.baidu.com/s/1sCfGQl7pLPxPIB9FmSlNcQ) key: aeo1

- Edit the file ```configs/DBCAN.py``` Make sure the paths of test datasets are right.

- Copy the path of the pretrained model && run
```bash tools/dist_test.sh configs/DBCAN.py [THE MODEL PATH] [GPU_NUMs]```

## For Train
Two simple steps to train your own model:

- To run the training code, please modify  ```configs/DBCAN.py``` to your own training data path. Make sure all the paths of Training set and Test set are right. 

- Run  ```bash tools/dist_train.sh configs/DBCAN.py [OUTPUT PATH] [GPU_NUMs]  ```


## Acknowledgement
The code of this project is modified from [MMOCR](https://github.com/open-mmlab/mmocr).


```[1] Kai Wang, Boris Babenko, and Serge Belongie. End-to-endscene text recognition. In ICCV, pages 1457–1464. IEEE,2011.```
