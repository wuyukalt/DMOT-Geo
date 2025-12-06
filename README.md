- Download the dataset [University-1652](https://github.com/layumi/University1652-Baseline) and [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark)
- Organize the data directory as follows:
```text
data/
├── train/
    ├──drone/
    └──satellite
└── test/
    ├──gallery_drone/
    ├──gallery_satellite/
    ├──query_drone/
    └──query_drone/
```
- Training
```text
python train_university.py
```
or
```text
python train_sues200.py
```

- Testing
```text
python predict_u1652.py
```
or
```text
python predict_s200.py
```
