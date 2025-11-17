"""
Few-Shot Learning on HAM10000 Skin Lesion Dataset using Prototypical Networks
Run this script to automatically train and evaluate a few-shot learning model.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import zipfile
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/kmader/skin-cancer-mnist-ham10000"
DATA_DIR = "./ham10000_data"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Few-shot settings
N_WAY = 5  # Number of classes per episode
K_SHOT = 5  # Number of support examples per class
N_QUERY = 15  # Number of query examples per class

print(f"Using device: {DEVICE}")

# ============================================================================
# STEP 1: DOWNLOAD HAM10000 DATASET
# ============================================================================

def download_ham10000():
    """Download HAM10000 dataset from Kaggle"""
    print("\n" + "="*60)
    print("DOWNLOADING HAM10000 DATASET")
    print("="*60)
    
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
        print(f"✓ Dataset already exists in {DATA_DIR}")
        return
    
    print("\nTo download HAM10000 from Kaggle, you need:")
    print("1. Kaggle account")
    print("2. Kaggle API key (kaggle.json)")
    print("\nSetup instructions:")
    print("   1. Go to kaggle.com/account")
    print("   2. Click 'Create New API Token'")
    print("   3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<You>\\.kaggle\\ (Windows)")
    
    try:
        # Try using Kaggle API
        import kaggle
        
        os.makedirs(DATA_DIR, exist_ok=True)
        
        print("\nDownloading HAM10000 dataset...")
        kaggle.api.dataset_download_files(
            'kmader/skin-cancer-mnist-ham10000',
            path=DATA_DIR,
            unzip=True
        )
        print("✓ Dataset downloaded successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nAlternative: Manual Download")
        print("1. Go to: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000")
        print("2. Download the dataset")
        print(f"3. Extract to: {DATA_DIR}")
        print("4. Re-run this script")
        exit(1)

# ============================================================================
# STEP 2: DATASET CLASS
# ============================================================================

class HAM10000Dataset(Dataset):
    """HAM10000 Skin Lesion Dataset"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Load metadata
        metadata_path = self.data_dir / 'HAM10000_metadata.csv'
        if not metadata_path.exists():
            # Try alternative names
            csv_files = list(self.data_dir.glob('*.csv'))
            if csv_files:
                metadata_path = csv_files[0]
            else:
                raise FileNotFoundError(f"No CSV file found in {data_dir}")
        
        self.metadata = pd.read_csv(metadata_path)
        
        # Find image directories
        img_dirs = []
        for d in ['HAM10000_images_part_1', 'HAM10000_images_part_2', 'HAM10000_images']:
            img_path = self.data_dir / d
            if img_path.exists():
                img_dirs.append(img_path)
        
        if not img_dirs:
            # Images might be directly in data_dir
            img_files = list(self.data_dir.glob('*.jpg'))
            if img_files:
                img_dirs = [self.data_dir]
        
        self.img_dirs = img_dirs
        
        # Split data
        total_samples = len(self.metadata)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        if split == 'train':
            self.metadata = self.metadata.iloc[:train_size]
        elif split == 'val':
            self.metadata = self.metadata.iloc[train_size:train_size+val_size]
        else:  # test
            self.metadata = self.metadata.iloc[train_size+val_size:]
        
        self.metadata = self.metadata.reset_index(drop=True)
        
        # Class mapping
        self.classes = sorted(self.metadata['dx'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"{split} set: {len(self.metadata)} images, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['image_id'] + '.jpg'
        
        # Find image in one of the directories
        img_path = None
        for img_dir in self.img_dirs:
            potential_path = img_dir / img_name
            if potential_path.exists():
                img_path = potential_path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[row['dx']]
        
        return image, label

# ============================================================================
# STEP 3: FEW-SHOT SAMPLER
# ============================================================================

class FewShotSampler:
    """Sample N-way K-shot episodes"""
    
    def __init__(self, dataset, n_way=5, k_shot=5, n_query=15):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        
        # Organize by class
        self.class_to_indices = {}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
    
    def sample_episode(self):
        """Sample one episode"""
        # Sample N classes
        episode_classes = np.random.choice(self.classes, size=self.n_way, replace=False)
        
        support_imgs, support_labels = [], []
        query_imgs, query_labels = [], []
        
        for class_idx, class_label in enumerate(episode_classes):
            indices = self.class_to_indices[class_label]
            
            # Sample K+Q examples
            n_samples = self.k_shot + self.n_query
            if len(indices) < n_samples:
                sampled = np.random.choice(indices, size=n_samples, replace=True)
            else:
                sampled = np.random.choice(indices, size=n_samples, replace=False)
            
            # Split into support and query
            for i, idx in enumerate(sampled):
                img, _ = self.dataset[idx]
                if i < self.k_shot:
                    support_imgs.append(img)
                    support_labels.append(class_idx)
                else:
                    query_imgs.append(img)
                    query_labels.append(class_idx)
        
        return (torch.stack(support_imgs), torch.tensor(support_labels),
                torch.stack(query_imgs), torch.tensor(query_labels))

# ============================================================================
# STEP 4: PROTOTYPICAL NETWORK MODEL
# ============================================================================

class PrototypicalNetwork(nn.Module):
    """Simple Prototypical Network with ResNet backbone"""
    
    def __init__(self, embedding_dim=512):
        super().__init__()
        
        # Use pretrained ResNet18 as backbone
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        """Extract features"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        return embeddings
    
    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        """Compute class prototypes"""
        prototypes = []
        for i in range(n_way):
            class_mask = support_labels == i
            prototype = support_embeddings[class_mask].mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)
    
    def predict(self, support_imgs, support_labels, query_imgs, n_way):
        """Predict query labels"""
        # Get embeddings
        support_emb = self.forward(support_imgs)
        query_emb = self.forward(query_imgs)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_emb, support_labels, n_way)
        
        # Compute distances
        distances = torch.cdist(query_emb, prototypes)
        
        # Predictions
        return torch.argmin(distances, dim=1)

# ============================================================================
# STEP 5: TRAINING
# ============================================================================

def train_few_shot(model, sampler, num_episodes=1000):
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    losses = []
    accuracies = []
    
    pbar = tqdm(range(num_episodes), desc='Training')
    for episode in pbar:
        # Sample episode
        sup_imgs, sup_labels, qry_imgs, qry_labels = sampler.sample_episode()
        
        sup_imgs = sup_imgs.to(DEVICE)
        sup_labels = sup_labels.to(DEVICE)
        qry_imgs = qry_imgs.to(DEVICE)
        qry_labels = qry_labels.to(DEVICE)
        
        # Forward
        optimizer.zero_grad()
        
        sup_emb = model(sup_imgs)
        qry_emb = model(qry_imgs)
        
        prototypes = model.compute_prototypes(sup_emb, sup_labels, N_WAY)
        distances = torch.cdist(qry_emb, prototypes)
        log_probs = F.log_softmax(-distances, dim=1)
        
        # Loss
        loss = F.nll_loss(log_probs, qry_labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        preds = torch.argmin(distances, dim=1)
        acc = (preds == qry_labels).float().mean().item()
        
        losses.append(loss.item())
        accuracies.append(acc)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    ax2.plot(accuracies)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves2.png', dpi=150)
    print("\n✓ Training curves saved to training_curves.png")
    
    return model

# ============================================================================
# STEP 6: EVALUATION
# ============================================================================

def evaluate_few_shot(model, sampler, num_episodes=200):
    """Evaluate on test episodes"""
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_episodes), desc='Evaluating'):
            sup_imgs, sup_labels, qry_imgs, qry_labels = sampler.sample_episode()
            
            sup_imgs = sup_imgs.to(DEVICE)
            sup_labels = sup_labels.to(DEVICE)
            qry_imgs = qry_imgs.to(DEVICE)
            qry_labels = qry_labels.to(DEVICE)
            
            preds = model.predict(sup_imgs, sup_labels, qry_imgs, N_WAY)
            acc = (preds == qry_labels).float().mean().item()
            accuracies.append(acc)
    
    # Statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    ci = 1.96 * std_acc / np.sqrt(num_episodes)
    
    print(f"\nResults ({N_WAY}-way {K_SHOT}-shot):")
    print(f"  Mean Accuracy: {mean_acc:.4f} ± {ci:.4f}")
    print(f"  Std Dev: {std_acc:.4f}")
    print(f"  Min: {min(accuracies):.4f}")
    print(f"  Max: {max(accuracies):.4f}")
    
    # Plot accuracy distribution
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.4f}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title(f'Test Accuracy Distribution ({num_episodes} episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('accuracy_distribution.png', dpi=150)
    print("✓ Accuracy distribution saved to accuracy_distribution.png")
    
    return mean_acc

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("FEW-SHOT LEARNING ON HAM10000")
    print("="*60)
    
    # Download dataset
    download_ham10000()
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = HAM10000Dataset(DATA_DIR, split='train', transform=transform)
    test_dataset = HAM10000Dataset(DATA_DIR, split='test', transform=transform)
    
    # Few-shot learning with Prototypical Networks
    print(f"\nStarting {N_WAY}-way {K_SHOT}-shot learning...")
    train_sampler = FewShotSampler(train_dataset, N_WAY, K_SHOT, N_QUERY)
    test_sampler = FewShotSampler(test_dataset, N_WAY, K_SHOT, N_QUERY)
    
    model = PrototypicalNetwork().to(DEVICE)
    model = train_few_shot(model, train_sampler, num_episodes=1000)
    
    # Save model
    torch.save(model.state_dict(), 'prototypical_model.pth')
    print("\n✓ Model saved to prototypical_model.pth")
    
    # Evaluate
    evaluate_few_shot(model, test_sampler, num_episodes=200)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()

