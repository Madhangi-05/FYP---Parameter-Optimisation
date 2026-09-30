"""Entry point for Stage 1 (dataset extraction/caching). Run this file
directly (`python run_build_dataset.py ...`), NOT `python -m src.data.build_dataset`
-- the latter makes Python treat build_dataset.py itself as __main__, which
breaks pickling CachedInstance for later loading by a different script (see
git history / the comment in build_dataset.py's module docstring area for
the full explanation). Importing it normally here keeps its classes on their
proper dotted import path, so the cached .pt files are loadable from
anywhere.
"""
from src.data.build_dataset import main

if __name__ == "__main__":
    main()
