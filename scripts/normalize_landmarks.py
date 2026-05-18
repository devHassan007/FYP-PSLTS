import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

def normalize_sequence(seq: np.ndarray, max_len: int = 300) -> np.ndarray:
  
    if len(seq) == 0:
        return np.zeros((max_len, seq.shape[1]))
    
    # Center: subtract mean of all non-zero points (avoid zeros from missing detections)
    non_zero_mask = np.any(seq != 0, axis=1)
    if np.any(non_zero_mask):
        mean = np.mean(seq[non_zero_mask], axis=0)
        seq -= mean
    
    # Scale: divide by std (avoid div-by-zero)
    std = np.std(seq, axis=0)
    std[std == 0] = 1.0
    seq /= std
    
    # Pad or truncate
    if len(seq) < max_len:
        padding = np.zeros((max_len - len(seq), seq.shape[1]))
        seq = np.vstack((seq, padding))
    elif len(seq) > max_len:
        seq = seq[:max_len]
    
    return seq.astype(np.float32)

def normalize_all(base_path: str = "psl_dataset"):
    raw_dir = Path(base_path) / "processed_landmarks"
    norm_dir = raw_dir / "normalized"
    norm_dir.mkdir(exist_ok=True, parents=True)
    
    for sentence_folder in raw_dir.iterdir():
        if sentence_folder.is_dir() and sentence_folder.name != "normalized":
            output_subdir = norm_dir / sentence_folder.name
            output_subdir.mkdir(exist_ok=True)
            
            npy_files = list(sentence_folder.glob("*.npy"))
            print(f"Normalizing {len(npy_files)} files in {sentence_folder.name}")
            
            for npy_path in tqdm(npy_files):
                seq = np.load(npy_path)
                norm_seq = normalize_sequence(seq)
                out_path = output_subdir / npy_path.name
                np.save(out_path, norm_seq)
                print(f"  Saved: {out_path} (shape: {norm_seq.shape})")

if __name__ == "__main__":
    normalize_all()