#!/usr/bin/env python3
import os
import io
import numpy as np
import webdataset as wds
import wids

# ─── parameters ─────────────────────────────────────────────────────────────
shard_rel   = "./datasets_preprocess/array_shard.tar"
index_rel   = "./datasets_preprocess/array_index.json"
num_rows    = 100
num_cols    = 50
read_index  = 42

# ensure output directory exists
os.makedirs(os.path.dirname(shard_rel), exist_ok=True)

print("=== WRITING SHARD ===")
with wds.TarWriter(shard_rel) as sink:
    for row_idx in range(num_rows):
        row = np.random.randn(num_cols).astype(np.float32)
        sink.write({
            "__key__": f"{row_idx:06d}",
            "row.npy": row,
        })
        if row_idx < 3 or row_idx == num_rows-1:
            print(f"  wrote sample {row_idx}, dtype={row.dtype}, shape={row.shape}")
print("wrote shard to", shard_rel)

print("\n=== INDEXING ===")
shard_abs = os.path.abspath(shard_rel)
print("absolute shard path:", shard_abs)
ret = os.system(f"widsindex create {shard_abs} -o {index_rel}")
print("widsindex exit code:", ret)
print("wrote index to", index_rel)

print("\n=== LOADING VIA WIDS ===")
dataset = wids.ShardListDataset(index_rel)
print("dataset length:", len(dataset))

def npy_from_blob(b, key_name):
    print(f"\n--- npy_from_blob for key '{key_name}' ---")
    print("  raw blob type:", type(b))
    # if it's already ndarray, just return it
    if isinstance(b, np.ndarray):
        print("  [blob is ndarray] shape:", b.shape, "dtype:", b.dtype)
        return b
    # if it's bytes
    if isinstance(b, (bytes, bytearray, memoryview)):
        raw = bytes(b)
        print(f"  [blob is raw bytes] len={len(raw)} first10={raw[:10]!r}")
        bio = io.BytesIO(raw)
        try:
            arr = np.load(bio, allow_pickle=True)
            print("  np.load bytes→ OK, got array shape", arr.shape, "dtype", arr.dtype)
            return arr
        except Exception as e:
            print("  np.load(bytes) FAILED:", type(e).__name__, e)
            raise

    # assume file-like
    try:
        pos = b.tell()
        hdr = b.read(10)
        b.seek(pos)
        print(f"  [blob is file-like] header bytes: {hdr!r}")
    except Exception as e:
        print("  could not peek file-like header:", type(e).__name__, e)

    try:
        arr = np.load(b, allow_pickle=True)
        print("  np.load file-like→ OK, got array shape", arr.shape, "dtype", arr.dtype)
        return arr
    except Exception as e:
        print("  np.load(file-like) FAILED:", type(e).__name__, e)
        raise

# now pull one sample
print(f"\nfetching sample[{read_index}]")
sample = dataset[read_index]
print("sample keys:", list(sample.keys())[:10], "… total", len(sample.keys()))

# try both the direct and the dot-prefixed key
for k in ("row.npy", ".row.npy"):
    if k in sample:
        print(f"\nattempting key '{k}'")
        blob = sample[k]
        row_i = npy_from_blob(blob, k)
        print(f"successfully loaded with key '{k}': shape={row_i.shape}")
        break
else:
    print("neither 'row.npy' nor '.row.npy' keys found in sample; available keys are:")
    print(list(sample.keys()))
