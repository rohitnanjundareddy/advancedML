# Advanced Machine Learning: Few-Shot and Transfer Learning for Medical Imaging

> **Improving Abnormality Detection with Data-Efficient and Generative Approaches**

This repository contains the implementation of few-shot learning and transfer learning techniques for medical abnormality detection, as presented in our CS512 Final Report. We demonstrate that combining pre-trained foundation models with few-shot adaptation offers a viable path toward data-efficient medical AI systems.

## 📄 Paper

**Few-shot and Transfer Learning for Deep Learning in Medical Imaging: Improving Abnormality Detection with Data-Efficient and Generative Approaches**

*Krish Patel & Rohit Nanjundareddy*  
*Department of Computer Science, University of Illinois Chicago*

[Read Full Paper](https://github.com/rohitnanjundareddy/advancedML/blob/main/CS512_Final_Report.pdf)

## 🎯 Key Results

- **48.5% accuracy** in 5-shot scenarios (vs. 38.4% for frozen features) using ResNet-18
- **50.5% accuracy** achieved by RAD-DINO (medical-specific ViT) in 10-shot scenarios
- **6-8% absolute improvement** from domain adaptation across all architectures
- **3-5% accuracy gains** from comprehensive data augmentation strategies

## 🏗️ Architecture Overview

Our pipeline consists of six integrated stages:

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Data Collection & Preprocessing (8-step pipeline) │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Stage 2: Data Augmentation (Geometric + Intensity + Rand)  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Stage 3: Synthetic Data Generation (Autoencoder/Diffusion) │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Stage 4: Transfer Learning (CNN: ResNet/DenseNet, ViT)     │
│           - RAD-DINO: Self-supervised ViT-B/14 on 882k CXRs │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Stage 5: Few-Shot Learning (Prototypical Networks)         │
│           - 1-shot, 5-shot, 10-shot episodic training       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Stage 6: Evaluation (Confusion Matrix, t-SNE, Metrics)     │
└─────────────────────────────────────────────────────────────┘
```

## 🔬 Methodology

### Prototypical Networks

We implement Prototypical Networks for few-shot classification:

For each class *c*, compute prototype:
```
c_k = (1/|S_k|) Σ f_φ(x_i)
```

Classification via softmax over distances:
```
p(y=k|x) = exp(-d(f_φ(x), c_k)) / Σ exp(-d(f_φ(x), c_k'))
```

### RAD-DINO: Self-Supervised Learning

RAD-DINO uses DINOv2-style self-distillation on medical images:

- **Student-Teacher Framework**: EMA-updated teacher network guides student
- **Multi-Crop Strategy**: 2 global crops (224×224) + 8 local crops (96×96)
- **Training Data**: 882k chest X-rays from multiple institutions
- **Architecture**: ViT-B/14 with 768-D embeddings

## 📊 Models Evaluated

| Model | Pre-training | Architecture | 1-Shot | 5-Shot | 10-Shot |
|-------|-------------|--------------|--------|--------|---------|
| ResNet-18 | ImageNet | CNN | 44.4% | 48.5% | 48.3% |
| DenseNet-121 | CheXNet | CNN | 45.8% | 47.5% | 46.5% |
| ViT-Base | ImageNet | Transformer | 36.2% | 39.9% | 39.6% |
| **RAD-DINO** | **Medical** | **ViT-B/14** | **46.5%** | **50.2%** | **50.5%** |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/rohitnanjundareddy/advancedML.git
cd advancedML
pip install -r requirements.txt
```

### Dataset Preparation

1. **Download CheXpert dataset**:
```bash
# Request access at: https://stanfordmlgroup.github.io/competitions/chexpert/
```

2. **Preprocess images** (8-stage pipeline):
```bash
python preprocess.py --input_dir /path/to/chexpert --output_dir ./data/processed
```

### Training

**Few-Shot Learning with ResNet-18**:
```bash
python train_few_shot.py \
    --model resnet18 \
    --n_way 3 \
    --k_shot 5 \
    --epochs 50 \
    --episodes_per_epoch 100
```

**Few-Shot Learning with RAD-DINO**:
```bash
python train_few_shot.py \
    --model rad_dino \
    --n_way 4 \
    --k_shot 5 \
    --pretrained_path ./models/rad_dino_teacher_encoder.pth \
    --freeze_backbone
```

**DINO Self-Supervised Pre-training** (optional):
```bash
python train_dino.py \
    --arch vit_base \
    --patch_size 14 \
    --global_crops_scale 0.4 1.0 \
    --local_crops_number 8 \
    --epochs 50
```

### Evaluation

```bash
python evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --n_way 3 \
    --k_shot 5 \
    --episodes 1000
```

## 📁 Repository Structure

```
advancedML/
├── data/
│   ├── preprocess.py          # 8-stage preprocessing pipeline
│   ├── augmentation.py        # Data augmentation strategies
│   └── dataset.py             # CheXpert dataset loader
├── models/
│   ├── prototypical.py        # Prototypical Networks
│   ├── backbones.py           # ResNet, DenseNet, ViT architectures
│   ├── rad_dino.py            # RAD-DINO implementation
│   └── dino_training.py       # Self-supervised DINO training
├── generators/
│   ├── autoencoder.py         # Autoencoder-based generation
│   └── diffusion.py           # Diffusion model generation
├── train_few_shot.py          # Few-shot training script
├── train_dino.py              # DINO pre-training script
├── evaluate.py                # Evaluation and metrics
├── utils/
│   ├── metrics.py             # Accuracy, confusion matrix
│   └── visualization.py       # t-SNE, UMAP, plots
├── configs/                   # Configuration files
├── notebooks/                 # Jupyter notebooks for analysis
└── requirements.txt
```

## 🔍 Key Features

### 8-Stage Preprocessing Pipeline
1. **Edge Detection** - Highlight anatomical boundaries
2. **Contrast Normalization** - Reduce scanner variability
3. **Feature Augmentation** - Emphasize subtle abnormalities
4. **Multi-Scale Enhancement** - Amplify textures at different scales
5. **Saliency Masking** - Focus on clinically relevant regions
6. **Histogram Equalization** - Improve low-contrast visibility
7. **Noise Reduction** - Remove artifacts
8. **Final Synthesis** - Create standardized representations

### Data Augmentation
- Geometric: Random rotations (±15°), flips, translations
- Intensity: Brightness, contrast, gamma corrections
- Advanced: RandAugment, MixUp strategies

### Synthetic Data Generation
- **Autoencoder**: Latent space perturbation for variants
- **Diffusion Models**: DDPM-style generation
- Limited to ~1/3 of real dataset size due to compute constraints

## 📈 Results Summary

### Domain Adaptation Impact
| Model | Frozen | Trained | Improvement |
|-------|--------|---------|-------------|
| ResNet-18 (5-shot) | 42.0% | 48.5% | +6.5% |
| DenseNet-121 (1-shot) | 38.2% | 45.8% | +7.6% |
| RAD-DINO (5-shot) | 43.5% | 50.2% | +6.7% |

### Architecture Insights
- **CNNs**: Strong baseline performance with ImageNet pre-training
- **ViT-Base**: Struggles without medical-specific pre-training (-8% vs ResNet)
- **RAD-DINO**: Best overall performance, demonstrating power of domain-specific ViT pre-training

## 🎓 Citation

```bibtex
@report{patel2024fewshot,
  title={Few-shot and Transfer Learning for Deep Learning in Medical Imaging: 
         Improving Abnormality Detection with Data-Efficient and Generative Approaches},
  author={Patel, Krish and Nanjundareddy, Rohit},
  institution={University of Illinois Chicago},
  year={2024},
  course={CS512}
}
```

## 🔮 Future Work

- [ ] **Synthetic Data Expansion**: Full GAN/Diffusion integration
- [ ] **MAML Implementation**: Model-Agnostic Meta-Learning
- [ ] **Vision-Language Models**: CLIP/MediCLIP fine-tuning
- [ ] **Multi-Label Classification**: Handle co-occurring pathologies
- [ ] **Uncertainty Quantification**: Monte Carlo dropout, ensembles
- [ ] **Clinical Validation**: Radiologist-in-the-loop studies
- [ ] **Cross-Institutional Testing**: Multi-hospital validation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Krish Patel** - [kpate446@uic.edu](mailto:kpate446@uic.edu)
- **Rohit Nanjundareddy** - [rnanj@uic.edu](mailto:rnanj@uic.edu)

## 🙏 Acknowledgments

- CheXpert dataset from Stanford ML Group
- RAD-DINO model from Microsoft Research
- Prototypical Networks framework
- UIC Computer Science Department

## 📚 References

1. Snell et al., "Prototypical networks for few-shot learning," NeurIPS 2017
2. Caron et al., "Emerging Properties in Self-Supervised Vision Transformers," 2021
3. Perez-García et al., "Exploring Scalable Medical Image Encoders Beyond Text Supervision," Nature MI 2025
4. Irvin et al., "CheXpert: A large chest radiograph dataset," AAAI 2019

---

**Note**: This research was conducted as part of CS512 (Advanced Machine Learning) at the University of Illinois Chicago. The models and code are provided for research and educational purposes.