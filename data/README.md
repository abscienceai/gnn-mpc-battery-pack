# Data Directory

Place BatteryML-format datasets here:

```
data/
└── processed/
    └── BatteryML/
        ├── CALCE/        # 13 LCO cells (CALCE dataset)
        │   ├── CS2_33.pkl
        │   └── ...
        ├── RWTH/         # 48 NMC cells (RWTH dataset)
        │   ├── RWTH_001.pkl
        │   └── ...
        ├── MATR/         # 180 LFP cells (MATR/Stanford dataset)
        │   ├── MATR_b1c0.pkl
        │   └── ...
        └── HUST/         # 77 LFP cells (HUST dataset)
            ├── HUST_1-1.pkl
            └── ...
```

## Download Instructions

All datasets are available through [BatteryML](https://github.com/microsoft/BatteryML).

```bash
git clone https://github.com/microsoft/BatteryML.git
cd BatteryML
python download_data.py --datasets CALCE RWTH MATR HUST
```

Move the processed pickle files to the directory structure above.

## Dataset Sizes

| Dataset | Cells | Chemistry | Source |
|---------|-------|-----------|--------|
| CALCE   | 13    | LCO       | https://calce.umd.edu/battery-data |
| RWTH    | 48    | NMC       | https://publications.rwth-aachen.de/record/818642 |
| MATR    | 180   | LFP       | https://data.matr.io/1/ |
| HUST    | 77    | LFP       | https://dx.doi.org/10.21227/6rc2-pd74 |


NASA (Randomized) and Oxford Deg1 datasets are used for generalisability/consistency checks only (Sections 5.x); no ECM extraction is performed on them, so no directory structure is required for reproducing the primary results.