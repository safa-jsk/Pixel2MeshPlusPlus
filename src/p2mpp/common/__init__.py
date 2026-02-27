"""
Common utilities shared across TF and PyTorch implementations.

Provides:
  - Project root path resolution
  - Artifact directory helpers
  - Canonical eval list path
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def get_project_root():
    """Return the absolute path to the project root (Pixel2MeshPlusPlus/)."""
    return PROJECT_ROOT

def get_artifacts_dir(subdir=""):
    """Return path to artifacts/<subdir>, creating it if needed."""
    path = os.path.join(PROJECT_ROOT, "artifacts", subdir)
    os.makedirs(path, exist_ok=True)
    return path

def get_configs_dir(design="designA"):
    """Return path to configs/<design>/."""
    return os.path.join(PROJECT_ROOT, "configs", design)

def get_assets_dir(subdir=""):
    """Return path to assets/<subdir>."""
    return os.path.join(PROJECT_ROOT, "assets", subdir)

def get_data_dir():
    """Return path to data/."""
    return os.path.join(PROJECT_ROOT, "data")

def get_eval_list_path():
    """Return the canonical evaluation list path."""
    return os.path.join(PROJECT_ROOT, "DesignA_CPU", "designA_eval_list.txt")
