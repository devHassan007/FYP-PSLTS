import torch
import os
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=225, hidden_size=64, num_classes=4, num_layers=2):  
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)  
        )
    
    def forward(self, x):
        lstm_out, (hn, _) = self.lstm(x)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)
        out = self.fc(hn)
        return out


def check_data_quality():
    """Diagnostic function to check if data is valid"""
    print("\n" + "="*60)
    print(" DATA QUALITY CHECK")
    print("="*60)
    
    try:
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
        
        print(f" Dataset loaded successfully")
        print(f" Training batches: {len(train_loader)}")
        print(f" Validation batches: {len(val_loader)}")
        print(f" Test batches: {len(test_loader)}")
        
        print("\n" + "-"*60)
        print("Checking first training batch...")
        for seqs, labels, texts in train_loader:
            print(f"\n Batch shape: {seqs.shape}")
            print(f" Labels in batch: {labels.tolist()}")
            print(f" Texts: {texts}")
            print(f"\nSequence Statistics:")
            print(f"  - Min value: {seqs.min():.4f}")
            print(f"  - Max value: {seqs.max():.4f}")
            print(f"  - Mean value: {seqs.mean():.4f}")
            print(f"  - Std value: {seqs.std():.4f}")
            
            has_nan      = torch.isnan(seqs).any().item()
            has_inf      = torch.isinf(seqs).any().item()
            all_zeros    = (seqs == 0).all().item()
            mostly_zeros = (seqs == 0).sum().item() / seqs.numel()
            
            print(f"\nData Quality Flags:")
            print(f"  - Contains NaN: {' YES (PROBLEM!)' if has_nan else ' No'}")
            print(f"  - Contains Inf: {' YES (PROBLEM!)' if has_inf else ' No'}")
            print(f"  - All zeros: {' YES (PROBLEM!)' if all_zeros else ' No'}")
            print(f"  - Percentage zeros: {mostly_zeros*100:.1f}%")
            
            if mostly_zeros > 0.9:
                print("    WARNING: More than 90% zeros - might indicate missing landmarks!")
            
            unique_labels = torch.unique(labels)
            print(f"\n✓ Unique labels in batch: {unique_labels.tolist()}")
            break
        
        print("\n" + "-"*60)
        print("Checking overall label distribution...")
        all_labels = []
        for seqs, labels, texts in train_loader:
            all_labels.extend(labels.tolist())
        
        from collections import Counter
        label_counts = Counter(all_labels)
        print(f"\nTraining set label distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  Class {label}: {count} samples")
        
        if len(label_counts) < 4:  # FIX: 3 → 4
            print(f"    WARNING: Only {len(label_counts)} classes found in training set (expected 4)!")
        
        print("\n" + "="*60)
        print(" Data quality check completed!")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"\n ERROR during data quality check: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def train():
    """Main training function with validation and progress tracking"""
    
    if not check_data_quality():
        print(" Data quality check failed. Please fix data issues before training.")
        return
    
    print("\n" + "="*60)
    print(" STARTING TRAINING")
    print("="*60 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
    
    model = LSTMClassifier(
        input_size=225,
        hidden_size=64,
        num_classes=4,   # FIX: 3 → 4
        num_layers=2
    ).to(device)
    
    print(f"\n Model Architecture:")
    print(model)
    
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Total parameters: {total_params:,}")
    print(f" Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # FIX: removed verbose=True — deprecated in newer PyTorch, causes a warning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=10,
        factor=0.5,
    )
    
    epochs              = 100
    best_val_loss       = float('inf')
    best_val_acc        = 0.0
    patience_counter    = 0
    early_stop_patience = 30
    
    print(f"\n  Training Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning rate: 0.001")
    print(f"  - Batch size: 4")
    print(f"  - Early stopping patience: {early_stop_patience}")
    print(f"  - Num classes: 4")
    
    print("\n" + "-"*60)
    print("Starting training loop...")
    print("-"*60 + "\n")
    
    for epoch in range(epochs):
        # ===== TRAINING =====
        model.train()
        train_loss = 0
        correct    = 0
        total      = 0
        
        for seqs, labels, _ in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc      = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # ===== VALIDATION =====
        model.eval()
        val_loss    = 0
        val_correct = 0
        val_total   = 0
        
        with torch.no_grad():
            for seqs, labels, _ in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                loss    = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total   += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc      = 100 * val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print LR reduction manually (replaces verbose=True)
        if current_lr != optimizer.param_groups[0]['lr']:
            print(f"      LR reduced to {current_lr:.6f}")
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:6.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:6.2f}% | "
              f"LR: {current_lr:.6f}")
        
        if avg_val_loss < best_val_loss and avg_val_loss > 0:
            best_val_loss    = avg_val_loss
            best_val_acc     = val_acc
            patience_counter = 0
            
            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss':           avg_train_loss,
                'val_loss':             avg_val_loss,
                'val_acc':              val_acc,
                'num_classes':          4,   # saved for reference
            }, "models/classifier_best.pth")
            print(f"      Saved best model (val_loss: {best_val_loss:.4f}, val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping triggered after {epoch+1} epochs")
            print(f"    No improvement for {early_stop_patience} epochs")
            break
    
    # ===== FINAL RESULTS =====
    print("\n" + "="*60)
    print(" TRAINING COMPLETED!")
    print("="*60)
    print(f"\n Best Results:")
    print(f"  - Best Validation Loss: {best_val_loss:.4f}")
    print(f"  - Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  - Model saved to: models/classifier_best.pth")
    
    # ===== TEST EVALUATION =====
    print("\n" + "-"*60)
    print(" Evaluating on test set...")
    print("-"*60)
    
    checkpoint = torch.load("models/classifier_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct    = 0
    test_total      = 0
    all_predictions = []
    all_labels      = []
    
    with torch.no_grad():
        for seqs, labels, _ in test_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs      = model(seqs)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total      += labels.size(0)
            test_correct    += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total if test_total > 0 else 0
    print(f"\n Test Accuracy: {test_acc:.2f}% ({test_correct}/{test_total})")
    
    if test_total > 0:
        from collections import defaultdict
        confusion = defaultdict(lambda: defaultdict(int))
        for true_label, pred_label in zip(all_labels, all_predictions):
            confusion[true_label][pred_label] += 1
        
        # FIX: confusion matrix now shows 4 classes (0-3)
        print("\n Confusion Matrix:")
        header = "          " + "".join(f"Pred {i}  " for i in range(4))
        print(header)
        for true_label in sorted(confusion.keys()):
            row = f"True {true_label}:  "
            for pred_label in range(4):   # FIX: range(3) → range(4)
                count = confusion[true_label][pred_label]
                row  += f"{count:5d}    "
            print(row)
    
    print("\n" + "="*60)
    print(" ALL DONE!")
    print("="*60 + "\n")


def predict_video(video_landmarks_path, model_path="models/classifier_best.pth"):
   
    import json
    import numpy as np

    with open("psl_dataset/labels/sentence_labels.json", 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    id_to_name = {v['class_id']: k       for k, v in labels.items()}
    id_to_urdu = {v['class_id']: v['urdu_text'] for k, v in labels.items()}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMClassifier(input_size=225, hidden_size=64, num_classes=4, num_layers=2)  # FIX: 3 → 4
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    landmarks = np.load(video_landmarks_path)
    landmarks = torch.from_numpy(landmarks).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output        = model(landmarks)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class  = predicted.item()
    class_name       = id_to_name[predicted_class]
    urdu_text        = id_to_urdu[predicted_class]
    confidence_score = confidence.item() * 100
    
    return predicted_class, class_name, urdu_text, confidence_score, probabilities[0].cpu().numpy()


if __name__ == "__main__":
    train()