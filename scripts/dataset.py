import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class PSLDataset(Dataset):
    def __init__(self, norm_dir: str, labels_file: str, augment=False):
       
        self.norm_dir = Path(norm_dir)
        self.augment = augment
        
        # Load labels
        with open(labels_file, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        
        # Map folder name to class_id and urdu_text
        self.class_map = {k: v['class_id'] for k, v in self.labels.items()}
        self.text_map = {k: v['urdu_text'] for k, v in self.labels.items()}
        
        # Collect all samples
        self.samples = []
        for sentence in self.class_map:
            folder = self.norm_dir / sentence
            if not folder.exists():
                print(f" Warning: Folder not found: {folder}")
                continue
                
            npy_files = list(folder.glob("*.npy"))
            print(f"Found {len(npy_files)} files in {sentence}/")
            
            for npy in npy_files:
                self.samples.append((
                    str(npy), 
                    self.class_map[sentence], 
                    self.text_map[sentence]
                ))
        
        print(f"Total samples loaded: {len(self.samples)}")
        
        # Print class distribution
        from collections import Counter
        label_counts = Counter([s[1] for s in self.samples])
        print(f"Class distribution:")
        for label, count in sorted(label_counts.items()):
            class_name = [k for k, v in self.class_map.items() if v == label][0]
            print(f"   Class {label} ({class_name}): {count} samples")
    
    def augment_sequence(self, seq):
      
        seq = seq.clone()
        
        # 1. Random Gaussian noise (50% chance)
        if np.random.rand() > 0.5:
            noise_level = np.random.uniform(0.01, 0.05)
            noise = torch.randn_like(seq) * noise_level
            seq = seq + noise
        
        # 2. Random time stretch (50% chance)
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.85, 1.15)  # 85% to 115% speed
            original_len = len(seq)
            new_len = int(original_len * factor)
            
            # Simple linear interpolation
            indices = torch.linspace(0, original_len - 1, new_len)
            indices_floor = indices.long()
            indices_ceil = torch.clamp(indices_floor + 1, max=original_len - 1)
            
            weight = (indices - indices_floor.float()).unsqueeze(1)
            seq = seq[indices_floor] * (1 - weight) + seq[indices_ceil] * weight
            
            # Pad or truncate back to 300
            if len(seq) < 300:
                padding = torch.zeros(300 - len(seq), seq.shape[1])
                seq = torch.vstack((seq, padding))
            else:
                seq = seq[:300]
        
        # 3. Random spatial scaling (30% chance)
        if np.random.rand() > 0.7:
            scale_factor = np.random.uniform(0.95, 1.05)
            seq = seq * scale_factor
        
        return seq
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label, text = self.samples[idx]
        
        # Load sequence
        seq = np.load(path)  # (max_len, 225)
        seq = torch.from_numpy(seq).float()
        
        # Apply augmentation if enabled
        if self.augment:
            seq = self.augment_sequence(seq)
        
        return seq, label, text


def get_dataloaders(batch_size=10, augment_train=True):
   
    # Load full dataset (no augmentation initially)
    full_dataset = PSLDataset(
        norm_dir="psl_dataset/processed_landmarks/normalized",
        labels_file="psl_dataset/labels/sentence_labels.json",
        augment=False  # We'll set augmentation per split
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\n Dataset Split:")
    print(f"   Training: {train_size} samples")
    print(f"   Validation: {val_size} samples")
    print(f"   Test: {test_size} samples")
    
    # Random split
    train_ds, val_ds, test_ds = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Enable augmentation for training set only
    if augment_train:
        # Create a wrapper that enables augmentation
        class AugmentedSubset(torch.utils.data.Subset):
            def __getitem__(self, idx):
                self.dataset.augment = True
                result = super().__getitem__(idx)
                self.dataset.augment = False
                return result
        
        # Wrap training subset with augmentation
        train_ds = AugmentedSubset(full_dataset, train_ds.indices)
        print(f" Data augmentation enabled for training set")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


# Test function to verify dataset
def test_dataset():
    print(" Testing dataset loading...\n")
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, augment_train=True)
    
    print("\n" + "="*60)
    print("Testing training batch (with augmentation):")
    print("="*60)
    
    for seqs, labels, texts in train_loader:
        print(f"Batch shape: {seqs.shape}")
        print(f"Labels: {labels}")
        print(f"Texts: {texts}")
        print(f"Sequence range: [{seqs.min():.3f}, {seqs.max():.3f}]")
        break
    
    print("\n" + "="*60)
    print("Testing validation batch (no augmentation):")
    print("="*60)
    
    for seqs, labels, texts in val_loader:
        print(f"Batch shape: {seqs.shape}")
        print(f"Labels: {labels}")
        print(f"Texts: {texts}")
        print(f"Sequence range: [{seqs.min():.3f}, {seqs.max():.3f}]")
        break
    
    print("\n Dataset test completed!")


if __name__ == "__main__":
    test_dataset()