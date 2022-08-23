[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=Xiefeng69.EpiGNN
[repo-url]: https://github.com/Xiefeng69/EpiGNN

# EpiGNN

[ECML-PKDD2022] The source codes and datasets for `EpiGNN: Exploring Spatial Transmission with Graph Neural Network for Regional Epidemic Forecasting`. Specifically, the codes are in the `\src`, while data is in the `\data`.

[![visitors][visitors-img]][repo-url]

## 1. Introduction

Epidemic forecasting is the key to effective control of epidemic transmission and helps the world mitigate the crisis that threatens public health. To better understand the transmission and evolution of epidemics, we propose EpiGNN, a graph neural network-based model for epidemic forecasting. Specifically, we design a transmission risk encoding module to characterize local and global spatial effects of regions in a epidemic process and incorporate them into the model. Meanwhile, we develop a Region-Aware Graph Learner (RAGL) that takes transmission risk, geographical dependencies, and temporal information into account to better explore spatial-temporal dependencies between regions and makes regions aware of related regions' epidemic situations. The RAGL can also combine with external resources, such as human mobility, to further improve prediction performance. Comprehensive experiments on five epidemic-related datasets (including influenza and COVID-19) demonstrate the effectiveness of our proposed method and show that EpiGNN outperforms state-of-the-art baselines by 9.48% in RMSE.

## 2. Datasets
### 2.1 Epidemic Statistics

The Influenza-related datasets are released by [Cola-GNN](https://github.com/amy-deng/colagnn) and the COVID-related data is publicly avaliable at [JHU-CSSE](https://github.com/CSSEGISandData/COVID-19).

### 2.2 Human Mobility

The graphs are formed using the movement data from facebook [Data For Good disease prevention maps](https://dataforgood.fb.com/docs/covid19/).

## 3. Quick Start

All programs are implemented using Python 3.8.5 and PyTorch 1.9.1 with CUDA 11.1 (1.9.1 cu111) in an Ubuntu server with an Nvidia Tesla K80 GPU. Install [Pytorch](https://pytorch.org/get-started/locally/).

```shell
cd EpiGNN
pip install -r requirements.txt
```

run the US-Region dataset as example:
```shell
python src/train.py --gpu 0 --lr 0.005 --horizon 5 --hidR 64 --hidA 64 --data region785 --sim_mat region-adj
```

### 3.1 Parameters

+ *hidR*: the hidden dimension of model.
+ *hidA*: the attention hidden dimension in GTR encoder.
+ *data*: the confirmed case data in the folder `\data`.
+ *sim_mat*: the adjacent matrix.
+ *lr*: learning rate.
+ *hw*: the look-back window of AutoRegresssive Component.
+ *batch*: batch size.
+ *epoch*: the number of epochs of traning process.
+ *patience*: we conduct early stop with fixed patience.

## More about EPIDEMICs

+ Seasonal influenza: [https://www.who.int/en/news-room/fact-sheets/detail/influenza-(seasonal)](https://www.who.int/en/news-room/fact-sheets/detail/influenza-(seasonal))
+ COVID-19 pandemic: [https://covid19.who.int/](https://covid19.who.int/)
+ The epidemic surveillance system of Lanzhou University: [http://covid-19.lzu.edu.cn/index.htm](http://covid-19.lzu.edu.cn/index.htm)

## Citation

```
@inproceedings{xie2022epignn,
  title={EpiGNN: Exploring Spatial Transmission with Graph Neural Network for Regional Epidemic Forecasting},
  author={Xie, Feng and Zhang, Zhong and Li, Liang and Zhou, Bin and Tan, Yusong},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  year={2022},
  organization={Springer}
}
```
