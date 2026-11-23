# Dataset Directory

- **Dataset:** House Price Prediction Dataset Vietnam - 2024
- **Source:** https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024
- **Expected local file:** `data/raw/vietnam_housing_dataset.csv` (`30,229` rows × `12` columns, gitignored)

If `data/raw/vietnam_housing_dataset.csv` is not present on a teammate's machine, download and extract it into `data/raw/`:

```bash
curl -sSL -o vietnam-housing-dataset-2024.zip \
  https://www.kaggle.com/api/v1/datasets/download/nguyentiennhan/vietnam-housing-dataset-2024
unzip -o vietnam-housing-dataset-2024.zip -d final-project/data/raw/
rm vietnam-housing-dataset-2024.zip
```

See [../docs/dataset-investigation.md](../docs/dataset-investigation.md) for full EDA findings, data quality traps, and binning benchmarks.
