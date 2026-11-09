"""Perturbation-as-action project: data, splits, frozen encoders, embeddings, decoders, evaluation."""
import os

# numba's OpenMP threading layer segfaults when PyTorch's libomp is loaded in the same process (macOS);
# the workqueue layer avoids it. Must be set before numba (scanpy, pecanpy, umap) is imported.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
