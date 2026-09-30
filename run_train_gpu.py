"""Entry point for Stage 3 (GPU training). Run `python run_build_dataset.py`
first to produce the cached dataset this reads from `data_cache/`.
"""
from src.quantum.train_gpu import main

if __name__ == "__main__":
    main()
