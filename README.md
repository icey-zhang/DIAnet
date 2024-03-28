# Distribution-aware Interactive Attention Network and Large-scale Cloud Recognition Benchmark on FY-4A Satellite Image

## Paper
⭐ This code will be released when the paper is accepted⭐ 

⭐ our [article](https://arxiv.org/abs/2401.03182) ⭐ 

## Citation

```
@misc{zhang2024multimodal,
      title={Multimodal Informative ViT: Information Aggregation and Distribution for Hyperspectral and LiDAR Classification}, 
      author={Jiaqing Zhang and Jie Lei and Weiying Xie and Geng Yang and Daixun Li and Yunsong Li and Karim Seghouane},
      year={2024},
      eprint={2401.03179},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Products

<p align="center"> <img src="Products/1.gif" width="30%">  <img src="Products\2.gif" width="30%">  <img src="Products\3.gif" width="30%"></p>

## Prepare the dataset
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
│   │   ├── 20200104
│   │   │   ├── FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200104003000_20200104003417_4000M_V0001.HDF
│   │   │   ├── FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200104003418_20200104003835_4000M_V0001.HDF
......
```



