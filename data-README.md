# Dataset Information

## Overview

This project uses **synthetically generated Criteo-style advertising click data** that mimics real-world patterns found in online advertising platforms.

## Dataset Specifications

- **Total Samples:** 100,000
- **Continuous Features:** 13 (int_1 to int_13)
- **Categorical Features:** 26 (cat_1 to cat_26)
- **Total Size:** 26.92 MB (CSV format)
- **Missing Values:** 5% in continuous, 2% in categorical

## Feature Distributions

### Continuous Features
- Power-law distribution: P(x) ∝ (x+1)^(-α), where α ∈ {1.5, 2.0}
- Range: 0-1000 for most features
- Intentional missing values for realism

### Categorical Features
- Zipfian distribution with varying cardinality:
  - cat_1 to cat_3: 20 unique values
  - cat_4 to cat_8: 100 unique values
  - cat_9 to cat_15: 500 unique values
  - cat_16 to cat_26: 2000 unique values

## Data Generation

**The dataset is generated automatically in the notebook** using fixed random seeds for reproducibility:
- `np.random.seed(42)`
- `torch.manual_seed(42)`

No external downloads required!

## Public Dataset Link

Since the data is synthetically generated, you can reproduce the exact dataset by running the notebook:

**Google Colab Notebook:** https://colab.research.google.com/drive/14XzXIt7hxeqtSzOP76wfodCQqF8Ai-Xq?usp=sharing

The notebook includes:
1. Complete data generation code (Cell 2)
2. Preprocessing pipeline (Cell 3)
3. Dataset statistics and visualization

## Usage

To generate the dataset locally:

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# See notebook Cell 2 for complete generation code
# Dataset will be saved as: criteo_production_100k.csv
```

## Citation

This dataset format is inspired by:
- Criteo Display Advertising Challenge (Kaggle)
- Real-world click-through rate prediction datasets

---

**Note:** For academic reproducibility, the complete generation code with all parameters is available in the main notebook.
