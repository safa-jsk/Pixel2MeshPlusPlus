"""
Path bootstrap for TF scripts (Design A).

Import this at the top of any script that needs to access
the TF modules and utils from their new location in src/p2mpp/tf/.

Usage (at top of script, before any p2mpp imports):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.p2mpp.tf import _bootstrap  # noqa: F401

Or simply call setup_tf_paths() from here.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the tf/ directory to sys.path so that `from modules.X import Y` and
# `from utils.X import Y` still work without changing every single import
# line in the TF codebase.  Also add project root for `from external.*`.
_tf_root = os.path.join(_PROJECT_ROOT, 'src', 'p2mpp', 'tf')

if _tf_root not in sys.path:
    sys.path.insert(0, _tf_root)

# Also ensure project root is in path (for configs, data, etc.)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def setup_tf_paths():
    """Explicitly set up sys.path for TF module resolution.
    
    Call this from any TF script before importing from modules.* or utils.*.
    """
    if _tf_root not in sys.path:
        sys.path.insert(0, _tf_root)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    return _PROJECT_ROOT
