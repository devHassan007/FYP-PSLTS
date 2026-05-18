
import torch
import numpy as np
import json
from pathlib import Path
from train_classifier import LSTMClassifier


def predict_video(video_landmarks_path, model_path="models/classifier_best.pth"):
    
    # Load class labels
    with open("psl_dataset/labels/sentence_labels.json", 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    # Create reverse mapping (class_id -> name)
    id_to_name = {v['class_id']: k for k, v in labels.items()}
    id_to_urdu = {v['class_id']: v['urdu_text'] for k, v in labels.items()}
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_size=225, hidden_size=64, num_classes=3, num_layers=2)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load landmarks
    landmarks = np.load(video_landmarks_path)
    landmarks = torch.from_numpy(landmarks).float().unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        output = model(landmarks)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = predicted.item()
    class_name = id_to_name[predicted_class]
    urdu_text = id_to_urdu[predicted_class]
    confidence_score = confidence.item() * 100
    
    return predicted_class, class_name, urdu_text, confidence_score, probabilities[0].cpu().numpy()


def test_on_folder(folder_path, model_path="models/classifier_best.pth"):
    """Test model on all .npy files in a folder"""
    
    folder = Path(folder_path)
    npy_files = list(folder.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {folder_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"Testing on {len(npy_files)} videos from: {folder.name}")
    print(f"{'='*70}\n")
    
    correct = 0
    total = 0
    
    for npy_file in npy_files:
        try:
            pred_class, class_name, urdu_text, confidence, probs = predict_video(str(npy_file), model_path)
            
            # Check if correct (based on folder name)
            true_class = folder.name
            is_correct = (class_name == true_class)
            
            if is_correct:
                correct += 1
            total += 1
            
            status = "True" if is_correct else "False"
            
            print(f"{status} {npy_file.name:30s} | Predicted: {class_name:8s} ({urdu_text}) | Confidence: {confidence:5.1f}%")
            print(f"   Probabilities: hello={probs[0]*100:.1f}%, please={probs[1]*100:.1f}%, thanks={probs[2]*100:.1f}%")
            
        except Exception as e:
            print(f" Error processing {npy_file.name}: {str(e)}")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n{'='*70}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
    print(f"{'='*70}\n")


def test_single_video(video_path, model_path="models/classifier_best.pth"):
    """Test model on a single video and show detailed results"""
    
    print(f"\n{'='*70}")
    print(f"Testing: {Path(video_path).name}")
    print(f"{'='*70}\n")
    
    try:
        pred_class, class_name, urdu_text, confidence, probs = predict_video(video_path, model_path)
        
        print(f"   Prediction Results:")
        print(f"   Class ID: {pred_class}")
        print(f"   Class Name: {class_name}")
        print(f"   Urdu Text: {urdu_text}")
        print(f"   Confidence: {confidence:.2f}%")
        print(f"\n Probabilities for each class:")
        print(f"   Class 0 (hello):  {probs[0]*100:6.2f}%")
        print(f"   Class 1 (please): {probs[1]*100:6.2f}%")
        print(f"   Class 2 (thanks): {probs[2]*100:6.2f}%")
        print(f"\n{'='*70}\n")
        
    except Exception as e:
        print(f" Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    print("\n Sign Language Classifier - Inference Test")
    print("="*70)
    
    # Check if model exists
    if not Path("models/classifier_best.pth").exists():
        print("   Model not found! Please train the model first:")
        print("   python scripts/train_classifier.py")
        sys.exit(1)
    
    print("\nChoose an option:")
    print("1. Test a single video")
    print("2. Test all videos in a folder")
    print("3. Test all classes")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        video_path = input("Enter path to .npy file: ").strip()
        test_single_video(video_path)
        
    elif choice == "2":
        folder_path = input("Enter folder path: ").strip()
        test_on_folder(folder_path)
        
    elif choice == "3":
        # Test all classes
        base_path = Path("psl_dataset/processed_landmarks/normalized")
        for class_folder in ["hello", "please", "thanks"]:
            folder = base_path / class_folder
            if folder.exists():
                test_on_folder(folder)
            else:
                print(f"Folder not found: {folder}")
    
    else:
        print(" Invalid choice!")