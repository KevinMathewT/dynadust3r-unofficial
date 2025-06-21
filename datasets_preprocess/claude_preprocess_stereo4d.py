#!/usr/bin/env python3
"""
Convert Stereo4D dataset to WebDataset format
Author: Script Generator
Date: 2025-01-26
"""

import os
import glob
import pandas as pd
import webdataset as wds
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np

# =============================================================================
# CONSTANTS - Configure these paths according to your setup
# =============================================================================

# Dataset paths
DATASET_ROOT = "/path/to/stereo4d-data"
LEFTEYE_DIR = "/path/to/stereo4d-data/lefteye-perspective/train" 
ANNO_DIR = "/path/to/stereo4d-data/annotations/train"
META_CSV = "/path/to/stereo4d-data/meta/stereo4d_id_to_time_and_fov_metadata.csv"
STATS_CSV = "/path/to/stereo4d-data/meta/stats.csv"

# Output settings
OUTPUT_DIR = "./stereo4d_webdataset"
SHARD_SIZE = 1000  # Number of samples per shard
MAX_SAMPLES = None  # Set to None for all samples, or number to limit

# Multiprocessing settings
NUM_WORKERS = min(8, cpu_count())

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sequence_list():
    """Get list of valid sequences from metadata"""
    print("Loading metadata...")
    
    # Read metadata CSV
    meta_df = pd.read_csv(
        META_CSV,
        header=0,
        names=["vid", "clipid", "timestamp", "start_yaw", "end_yaw", "start_tilt", "end_tilt"],
    )
    
    # Read stats CSV and filter
    stats_df = pd.read_csv(STATS_CSV, skipinitialspace=True)
    stats_df = stats_df[stats_df["displacement_percentage_50"] > 0.10]
    stats_df = stats_df[stats_df["d_frame"] > 5 * 16]
    
    if MAX_SAMPLES is not None:
        stats_df = stats_df.sample(n=min(len(stats_df), MAX_SAMPLES), random_state=42)
    
    sequence_list = []
    
    for _, r in stats_df.iterrows():
        vid, cid = r["ytid"], r["clipid"]
        ts = meta_df.loc[(meta_df.vid == vid) & (meta_df.clipid == cid), "timestamp"]
        
        if ts.empty:
            continue
            
        seq = f"{vid}_{int(ts.values[0])}"
        
        # Check if both video and annotation files exist
        video_path = os.path.join(LEFTEYE_DIR, f"{seq}-left_rectified.mp4")
        anno_path = os.path.join(ANNO_DIR, f"{seq}.npz")
        
        if os.path.exists(video_path) and os.path.exists(anno_path):
            sequence_list.append({
                'seq': seq,
                'video_path': video_path,
                'anno_path': anno_path,
                'd_frame': int(r["d_frame"])
            })
    
    print(f"Found {len(sequence_list)} valid sequences")
    return sequence_list

def read_file_as_bytes(file_path):
    """Read file and return as bytes"""
    with open(file_path, 'rb') as f:
        return f.read()

def process_sequence_batch(args):
    """Process a batch of sequences and write to a shard"""
    sequences, shard_idx = args
    shard_path = os.path.join(OUTPUT_DIR, f"stereo4d_{shard_idx:06d}.tar")
    
    try:
        with wds.TarWriter(shard_path) as writer:
            for seq_info in sequences:
                seq = seq_info['seq']
                video_path = seq_info['video_path']
                anno_path = seq_info['anno_path']
                
                # Read files as bytes
                video_bytes = read_file_as_bytes(video_path)
                anno_bytes = read_file_as_bytes(anno_path)
                
                # Create sample for WebDataset
                sample = {
                    "__key__": seq,
                    "video.mp4": video_bytes,
                    "anno.npz": anno_bytes,
                    "metadata.json": {
                        "seq": seq,
                        "d_frame": seq_info['d_frame'],
                        "video_size": len(video_bytes),
                        "anno_size": len(anno_bytes)
                    }
                }
                
                writer.write(sample)
        
        return f"Successfully wrote shard {shard_idx} with {len(sequences)} sequences"
    
    except Exception as e:
        return f"Error writing shard {shard_idx}: {str(e)}"

# =============================================================================
# MAIN CONVERSION FUNCTION
# =============================================================================

def convert_to_webdataset():
    """Convert Stereo4D dataset to WebDataset format using multiprocessing"""
    print("Starting Stereo4D to WebDataset conversion...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get sequence list
    sequence_list = get_sequence_list()
    
    if not sequence_list:
        print("No valid sequences found!")
        return
    
    # Split sequences into batches for sharding
    sequence_batches = []
    for i in range(0, len(sequence_list), SHARD_SIZE):
        batch = sequence_list[i:i + SHARD_SIZE]
        sequence_batches.append((batch, i // SHARD_SIZE))
    
    print(f"Creating {len(sequence_batches)} shards with up to {SHARD_SIZE} sequences each")
    print(f"Using {NUM_WORKERS} workers for parallel processing")
    
    # Process batches in parallel
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_sequence_batch, sequence_batches),
            total=len(sequence_batches),
            desc="Writing shards"
        ))
    
    # Print results
    for result in results:
        print(result)
    
    print(f"Conversion complete! Shards saved to: {OUTPUT_DIR}")

# =============================================================================
# VERIFICATION FUNCTION
# =============================================================================

def verify_webdataset():
    """Verify the created WebDataset by reading all keys"""
    print("\n" + "="*50)
    print("VERIFICATION: Reading all keys from WebDataset...")
    print("="*50)
    
    # Find all shard files
    shard_pattern = os.path.join(OUTPUT_DIR, "stereo4d_*.tar")
    shard_files = sorted(glob.glob(shard_pattern))
    
    if not shard_files:
        print("ERROR: No shard files found!")
        return False
    
    print(f"Found {len(shard_files)} shard files")
    
    total_samples = 0
    errors = []
    
    for shard_file in shard_files:
        try:
            print(f"Checking shard: {os.path.basename(shard_file)}")
            
            # Create dataset from this shard
            dataset = wds.WebDataset([shard_file]).decode()
            
            shard_samples = 0
            for sample in dataset:
                try:
                    # Check required keys
                    required_keys = ['__key__', 'video.mp4', 'anno.npz', 'metadata.json']
                    for key in required_keys:
                        if key not in sample:
                            errors.append(f"Missing key '{key}' in sample {sample.get('__key__', 'unknown')}")
                    
                    # Verify data types
                    if isinstance(sample.get('video.mp4'), bytes):
                        video_size = len(sample['video.mp4'])
                    else:
                        errors.append(f"video.mp4 is not bytes for {sample['__key__']}")
                        
                    if isinstance(sample.get('anno.npz'), bytes):
                        anno_size = len(sample['anno.npz'])
                    else:
                        errors.append(f"anno.npz is not bytes for {sample['__key__']}")
                    
                    shard_samples += 1
                    
                except Exception as e:
                    errors.append(f"Error processing sample in {shard_file}: {str(e)}")
            
            total_samples += shard_samples
            print(f"  ✓ {shard_samples} samples verified")
            
        except Exception as e:
            errors.append(f"Error reading shard {shard_file}: {str(e)}")
    
    print(f"\nVerification complete!")
    print(f"Total samples verified: {total_samples}")
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    else:
        print("✅ All samples verified successfully!")
        return True

# =============================================================================
# EXAMPLE USAGE FUNCTION
# =============================================================================

def test_reading_samples():
    """Test reading a few samples from the WebDataset"""
    print("\n" + "="*50)
    print("TEST: Reading sample data from WebDataset...")
    print("="*50)
    
    # Create dataset
    shard_pattern = os.path.join(OUTPUT_DIR, "stereo4d_*.tar")
    dataset = wds.WebDataset(shard_pattern).decode()
    
    # Read first 3 samples
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
            
        print(f"\nSample {i+1}:")
        print(f"  Key: {sample['__key__']}")
        print(f"  Video size: {len(sample['video.mp4'])} bytes")
        print(f"  Anno size: {len(sample['anno.npz'])} bytes")
        print(f"  Metadata: {sample['metadata.json']}")
        
        # Try to load the NPZ to verify it's valid
        try:
            import io
            npz_data = np.load(io.BytesIO(sample['anno.npz']))
            print(f"  NPZ keys: {list(npz_data.keys())}")
        except Exception as e:
            print(f"  ❌ Error loading NPZ: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Stereo4D to WebDataset Converter")
    print("="*50)
    
    # Convert dataset
    convert_to_webdataset()
    
    # Verify the conversion
    verification_success = verify_webdataset()
    
    # Test reading samples
    if verification_success:
        test_reading_samples()
    
    print("\n🎉 Script completed!")