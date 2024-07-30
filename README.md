# Distribution-aware Interactive Attention Network and Large-scale Cloud Recognition Benchmark on FY-4A Satellite Image

## Paper 

⭐ our [article](https://arxiv.org/abs/2401.03182) ⭐ 

## Citation

```
@article{zhang2024distribution,
  title={Distribution-aware Interactive Attention Network and Large-scale Cloud Recognition Benchmark on FY-4A Satellite Image},
  author={Zhang, Jiaqing and Lei, Jie and Xie, Weiying and Jiang, Kai and Cao, Mingxiang and Li, Yunsong},
  journal={arXiv preprint arXiv:2401.03182},
  year={2024}
}
```
## Products

<p align="center"> <img src="Products/1.gif" width="30%">  <img src="Products\2.gif" width="30%">  <img src="Products\3.gif" width="30%"></p>

## Usage of the code for FYH dataset

### Prepare the dataset 
Download the FY4A L1 dataset for [FY4A](http://satellite.nsmc.org.cn/portalsite/Data/Satellite.aspx)

Download the Himawari dataset for [Himawari](http://www.jma-net.go.jp/msc/en/)

```python
├── fydatahimawari
│   ├── Himawari
......
│   │   ├── 202005
│   │   │   ├── 05
│   │   │   │   ├── 05
│   │   │   │   │   ├──NC_H08_20200505_0500_L2CLP010_FLDK.02401_02401.nc
│   │   │   │   │   ├──NC_H08_20200505_0510_L2CLP010_FLDK.02401_02401.nc
│   │   │   │   │   ├──NC_H08_20200505_0520_L2CLP010_FLDK.02401_02401.nc
......

│   ├── FY4A
......

│   │   ├── 20200104
│   │   │   ├── FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200104003000_20200104003417_4000M_V0001.HDF
│   │   │   ├── FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200104003418_20200104003835_4000M_V0001.HDF
......
```

### Split the dataset

```python
python split.py
```
### Train the model

```python
python tools/train.py --cfg experiments/cityscapes/seg_hrnet_AWCA_PSNL_z_w48_train_100x100_sgd_lr1e-2_wd5e-4_bs_8_epoch100.yaml
```
### Test the production

```python
python tools/test_production.py
```
## The generation validation of the SPARCS dataset

### Prepare the dataset
Download the Cloud detection SPARCS dataset 

### Split the dataset

```python
python split.py
```

### Train the model

```python
python tools/train.py  --cfg experiments/cityscapes/seg_hrnet_AWCA_PSNL_z_w48_train_200x200_sgd_lr1e-4_wd5e-4_bs_6_epoch100_sparcs.yaml
```
### Test the model

```python
python tools/test.py  --cfg experiments/cityscapes/seg_hrnet_AWCA_PSNL_z_w48_train_200x200_sgd_lr1e-4_wd5e-4_bs_6_epoch100_sparcs.yaml
```

