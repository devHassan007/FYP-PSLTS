# Data Quality Diagnostic Tool

import numpy as np
from pathlib import Path
import json


def check_normalized_data(base_path="psl_dataset"):
    
    print("="*70)
    print("PSL DATASET QUALITY DIAGNOSTIC")
    print("="*70)
    
    norm_dir = Path(base_path) / "processed_landmarks" / "normalized"
    labels_file = Path(base_path) / "labels" / "sentence_labels.json"
    
    print(f"\n Checking directory structure...")
    if not norm_dir.exists():
        print(f"ERROR: Normalized directory not found: {norm_dir}")
        return False
    else:
        print(f"Found: {norm_dir}")
    
    if not labels_file.exists():
        print(f"ERROR: Labels file not found: {labels_file}")
        return False
    else:
        print(f"Found: {labels_file}")
    
    # Load labels
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    print(f"\n Expected classes:")
    for name, info in labels.items():
        print(f"   {info['class_id']}: {name} ({info['urdu_text']})")
    
    # Check each class folder
    print(f"\n" + "="*70)
    print(" ANALYZING EACH CLASS")
    print("="*70)
    
    all_issues = []
    total_files = 0
    class_stats = {}
    
    for class_name in labels.keys():
        class_folder = norm_dir / class_name
        
        print(f"\n Class: {class_name}")
        print("-" * 50)
        
        if not class_folder.exists():
            print(f"   Folder not found: {class_folder}")
            all_issues.append(f"Missing folder: {class_name}")
            continue
        
        # Get all .npy files
        npy_files = list(class_folder.glob("*.npy"))
        print(f"    Files found: {len(npy_files)}")
        total_files += len(npy_files)
        
        if len(npy_files) == 0:
            print(f"     WARNING: No .npy files in {class_name}")
            all_issues.append(f"No files in {class_name}")
            continue
        
        # Analyze each file
        file_stats = {
            'shapes': [],
            'has_nan': 0,
            'has_inf': 0,
            'all_zeros': 0,
            'mostly_zeros': 0,
            'min_vals': [],
            'max_vals': [],
            'mean_vals': [],
        }
        
        for npy_file in npy_files:
            try:
                data = np.load(npy_file)
                
                # Check shape
                file_stats['shapes'].append(data.shape)
                
                # Check for NaN
                if np.isnan(data).any():
                    file_stats['has_nan'] += 1
                    all_issues.append(f"NaN in {class_name}/{npy_file.name}")
                
                # Check for Inf
                if np.isinf(data).any():
                    file_stats['has_inf'] += 1
                    all_issues.append(f"Inf in {class_name}/{npy_file.name}")
                
                # Check for all zeros
                if (data == 0).all():
                    file_stats['all_zeros'] += 1
                    all_issues.append(f"All zeros in {class_name}/{npy_file.name}")
                
                # Check for mostly zeros (>90%)
                zero_percentage = (data == 0).sum() / data.size
                if zero_percentage > 0.9:
                    file_stats['mostly_zeros'] += 1
                    all_issues.append(f"Mostly zeros ({zero_percentage*100:.1f}%) in {class_name}/{npy_file.name}")
                
                # Statistics
                file_stats['min_vals'].append(data.min())
                file_stats['max_vals'].append(data.max())
                file_stats['mean_vals'].append(data.mean())
                
            except Exception as e:
                print(f"    Error loading {npy_file.name}: {str(e)}")
                all_issues.append(f"Load error: {class_name}/{npy_file.name}")
        
        # Print statistics
        if file_stats['shapes']:
            unique_shapes = set(file_stats['shapes'])
            print(f"    Data shapes: {unique_shapes}")
            
            if len(unique_shapes) > 1:
                print(f"     WARNING: Inconsistent shapes detected!")
                all_issues.append(f"Inconsistent shapes in {class_name}")
            
            print(f"    Value range: [{np.min(file_stats['min_vals']):.3f}, {np.max(file_stats['max_vals']):.3f}]")
            print(f"    Mean value: {np.mean(file_stats['mean_vals']):.3f}")
            
            if file_stats['has_nan'] > 0:
                print(f"    Files with NaN: {file_stats['has_nan']}")
            
            if file_stats['has_inf'] > 0:
                print(f"    Files with Inf: {file_stats['has_inf']}")
            
            if file_stats['all_zeros'] > 0:
                print(f"     Files with all zeros: {file_stats['all_zeros']}")
            
            if file_stats['mostly_zeros'] > 0:
                print(f"     Files mostly zeros (>90%): {file_stats['mostly_zeros']}")
        
        class_stats[class_name] = {
            'file_count': len(npy_files),
            'issues': file_stats['has_nan'] + file_stats['has_inf'] + file_stats['all_zeros']
        }
    
    # Summary
    print(f"\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    print(f"\n Total files analyzed: {total_files}")
    print(f"\n Files per class:")
    for class_name, stats in class_stats.items():
        status = "GOOD" if stats['issues'] == 0 else "NOT GOOD "
        print(f"   {status} {class_name}: {stats['file_count']} files ({stats['issues']} issues)")
    
    # Check for class imbalance
    if class_stats:
        file_counts = [stats['file_count'] for stats in class_stats.values()]
        min_count = min(file_counts)
        max_count = max(file_counts)
        
        if max_count > min_count * 1.5:
            print(f"\n WARNING: Class imbalance detected!")
            print(f"   Range: {min_count} to {max_count} files per class")
            print(f"   Consider collecting more data for underrepresented classes")
        else:
            print(f"\n Classes are reasonably balanced ({min_count}-{max_count} files)")
    
    # Dataset size warning
    if total_files < 30:
        print(f"\n  WARNING: Very small dataset ({total_files} total files)")
        print(f"   Recommended: At least 50-100 files per class for good results")
        print(f"   Current: ~{total_files // len(class_stats)} files per class")
    
    # List all issues
    if all_issues:
        print(f"\n" + "="*70)
        print(f"  ISSUES FOUND ({len(all_issues)} total)")
        print("="*70)
        for i, issue in enumerate(all_issues[:20], 1):  # Show first 20
            print(f"   {i}. {issue}")
        
        if len(all_issues) > 20:
            print(f"   ... and {len(all_issues) - 20} more issues")
    else:
        print(f"\n No critical issues found!")
    
    print("\n" + "="*70)
    
    if len(all_issues) == 0 and total_files >= 20:
        print(" DATA QUALITY: GOOD - Ready for training!")
    elif len(all_issues) > 0 and total_files >= 20:
        print("  DATA QUALITY: ACCEPTABLE - Some issues found but can proceed")
    else:
        print(" DATA QUALITY: POOR - Need more data or fixes before training")
    
    print("="*70 + "\n")
    
    return len(all_issues) == 0


def check_raw_videos(base_path="psl_dataset"):
    """Check raw video files"""
    print("\n" + "="*70)
    print("📹 CHECKING RAW VIDEOS")
    print("="*70)
    
    raw_dir = Path(base_path) / "raw_videos"
    
    if not raw_dir.exists():
        print(f" Raw videos directory not found: {raw_dir}")
        return
    
    for class_folder in raw_dir.iterdir():
        if class_folder.is_dir():
            videos = list(class_folder.glob("*.mp4")) + list(class_folder.glob("*.avi"))
            print(f" {class_folder.name}: {len(videos)} videos")


if __name__ == "__main__":
    print("\n")
    
    # Check raw videos
    check_raw_videos()
    
    # Check normalized data
    is_good = check_normalized_data()
    
    print("\n RECOMMENDATIONS:")
    print("-" * 70)
    
    if is_good:
        print("1.  Your data looks good! Proceed with training.")
        print("2. Run: python train_classifier.py")
    else:
        print("1. Fix the issues listed above")
        print("2. If you have too few videos, record more samples")
        print("3. Re-run landmark extraction if needed")
        print("4. Then run: python train_classifier.py")
    
    print("-" * 70 + "\n")