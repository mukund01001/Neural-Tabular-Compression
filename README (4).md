# Neural Tabular Data Compression Using Deep Autoencoders

**Author:** Mukund Rathi  
**Roll Number:** 2023BEC0051  
**Institution:** Indian Institute of Information Technology Kottayam  
**Course:** CSE 311 - Artificial Intelligence  
**Date:** November 2025

---

## 🎯 Project Overview

This project implements a deep learning approach to compress tabular data using autoencoder neural networks with learned embeddings and post-training quantization. The method achieves **16.2× compression ratio with 99.3% reconstruction accuracy** on advertising click data, significantly outperforming classical methods like gzip (10.8×) and PCA (2.6×).

### Key Results
- **Compression Ratio:** Up to 16.2× (exceeds 8-15× target)
- **Reconstruction Accuracy:** >99% across all configurations
- **Performance:** 50% better compression than gzip with minimal quality loss

---

## 📊 Results Summary

| Method          | Compression | Accuracy | Type     |
|-----------------|-------------|----------|----------|
| Gzip            | 10.8×      | 100.0%   | Lossless |
| PCA (k=5)       | 2.6×       | 68.7%    | Lossy    |
| **AE k=128**    | **12.5×**  | **99.5%**| Lossy    |
| **AE k=64**     | **15.0×**  | **99.4%**| Lossy    |
| **AE k=32**     | **16.2×**  | **99.3%**| Lossy    |

---

## 🗂️ Repository Structure

```
neural-tabular-compression/
├── README.md                          # This file
├── report.pdf                         # Complete project report
├── notebooks/
│   └── neural_compression.ipynb      # Main Colab notebook
├── figures/
│   ├── architecture.png              # Model architecture
│   ├── results_comparison.png        # Performance charts
│   └── workflow.png                  # Pipeline diagram
├── data/
│   └── README.md                     # Dataset information
├── models/
│   └── README.md                     # Pretrained models info
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
Click here to run the complete notebook with pretrained models:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14XzXIt7hxeqtSzOP76wfodCQqF8Ai-Xq?usp=sharing)

**No setup required!** The notebook includes:
- Automatic dataset generation
- Pretrained model loading (instant demo)
- Full training pipeline (2 hours on GPU)
- All visualizations and analysis

### Option 2: Local Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/neural-tabular-compression.git
cd neural-tabular-compression

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/neural_compression.ipynb
```

---

## 📦 Dataset

The project uses synthetically generated Criteo-style advertising click data:
- **Samples:** 100,000
- **Features:** 13 continuous + 26 categorical
- **Size:** 26.92 MB (CSV)
- **Distribution:** Power-law for continuous, Zipfian for categorical

**Dataset is generated automatically** in the notebook. No external downloads required.

For reproducibility, the exact generation code with fixed random seeds (42) is included.

---

## 🏗️ Architecture

Our autoencoder architecture:

```
Input (173D)
    ↓
Embeddings (10×16D) + Continuous (13D)
    ↓
Encoder: 173 → 512 → 256 → k
    ↓
Bottleneck: k ∈ {32, 64, 128}
    ↓
Decoder: k → 256 → 512
    ↓
Outputs: 10 categorical heads + 1 continuous head
```

**Post-processing:**
1. 8-bit quantization of latent codes
2. LZMA entropy coding
3. Model weights stored separately (~1.5 MB)

---

## 🔬 Methodology

### Training Details
- **Framework:** PyTorch 2.0+
- **Device:** NVIDIA T4 GPU (Google Colab)
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 50 per configuration
- **Batch size:** 256
- **Loss:** Cross-entropy (categorical) + MSE (continuous)

### Evaluation Metrics
- Compression ratio: Original size / Compressed size
- Categorical accuracy: Correct predictions / Total predictions
- Continuous MSE: Mean squared error on numerical features
- Efficiency: Actual vs. theoretical compression (entropy-based)

---

## 📈 Key Findings

1. **Neural compression achieves 50% better compression than gzip** (16.2× vs. 10.8×) with only 0.68% accuracy loss
2. **All configurations maintain >99% accuracy** vs. PCA's 68.7%
3. **Near-optimal compression efficiency** (>100% vs. Shannon entropy bound)
4. **Tunable compression-accuracy trade-off** via bottleneck size

---

## 📁 Files Description

### Notebooks
- `neural_compression.ipynb`: Complete implementation with training, evaluation, and visualization

### Figures
- `architecture.png`: Neural autoencoder architecture diagram
- `results_comparison.png`: Compression vs. accuracy comparison chart
- `workflow.png`: Data processing pipeline

### Models
Pretrained models stored in Google Drive (linked in notebook):
- `autoencoder_k32.pt`
- `autoencoder_k64.pt`
- `autoencoder_k128.pt`

---

## 🛠️ Requirements

```
Python >= 3.10
PyTorch >= 2.0
pandas >= 1.5
numpy >= 1.23
scikit-learn >= 1.2
matplotlib >= 3.6
```

See `requirements.txt` for complete list.

---

## 📄 Report

The complete project report (LaTeX source and PDF) is available in the repository:
- Format: IEEE-style academic report
- Sections: Abstract, Introduction, Methodology, Results, Discussion, Conclusion
- Length: ~15 pages
- Figures: 4 (architecture, results, entropy analysis, workflow)

---

## 🎓 Academic Context

**Course:** CSE 311 - Artificial Intelligence  
**Institution:** IIIT Kottayam  
**Submission Date:** November 2025  

This project demonstrates the application of deep learning to data compression, validating that neural methods can learn data-specific patterns superior to general-purpose algorithms.

---

## 🔗 Links

- **Colab Notebook:** [Open in Colab](https://colab.research.google.com/drive/14XzXIt7hxeqtSzOP76wfodCQqF8Ai-Xq?usp=sharing)
- **Report PDF:** [View Report](./report.pdf)
- **Pretrained Models:** Available in Google Drive (see notebook)

---

## 📧 Contact

**Mukund Rathi**  
Roll: 2023BEC0051  
Email: mukund23bec51@iiitkottayam.ac.in  
Department of Electronics and Communication Engineering  
Indian Institute of Information Technology Kottayam

---

## 📜 License

This project is submitted as part of academic coursework for CSE 311 at IIIT Kottayam.

---

## 🙏 Acknowledgments

- Course Instructor: CSE 311 Faculty, IIIT Kottayam
- Framework: PyTorch Team
- Platform: Google Colab

---

## 📚 References

1. Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*.
2. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational Bayes. *arXiv preprint*.
3. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*.

---

**Last Updated:** November 13, 2025