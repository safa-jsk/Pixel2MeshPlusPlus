import os as _os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import ops

# Resolve project root so .so path works regardless of CWD
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))

# Try to load the compiled op first (GPU/fast path)
_nn_mod = None
_use_custom_op = False

_so_search_paths = [
    _os.path.join(_PROJECT_ROOT, 'external', 'tf_ops', 'prebuilt', 'tf_nndistance_so.so'),
    _os.path.join(_PROJECT_ROOT, 'external', 'tf_nndistance_so.so'),  # legacy
    './external/tf_nndistance_so.so',                                   # Docker CWD
    './external/tf_ops/prebuilt/tf_nndistance_so.so',
]

for _so_path in _so_search_paths:
    try:
        _nn_mod = tf.load_op_library(_so_path)
        _use_custom_op = True
        print("[INFO] Loaded custom NNDistance op: {}".format(_so_path))
        break
    except Exception:
        continue

if not _use_custom_op:
    print("[WARN] Failed to load custom NNDistance op from any path. Falling back to CPU implementation.")
    _use_custom_op = False

if not _use_custom_op:
    # CPU fallback (pure TF/Python)
    import importlib, sys
    _cpu_mod_path = _os.path.join(_PROJECT_ROOT, 'external', 'tf_ops', 'src')
    if _cpu_mod_path not in sys.path:
        sys.path.insert(0, _cpu_mod_path)
    from tf_nndistance_cpu import nn_distance_cpu as _cpu_nn_distance


def nn_distance(xyz1, xyz2):
    """
    Computes the distance of nearest neighbors for a pair of point clouds
    input: xyz1: (batch_size,#points_1,3)  the first point cloud
    input: xyz2: (batch_size,#points_2,3)  the second point cloud
    output: dist1: (batch_size,#point_1)   distance from first to second
    output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
    output: dist2: (batch_size,#point_2)   distance from second to first
    output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
    """
    # NOTE: Your original code wrapped xyz1/xyz2 with expand_dims.
    # Keep behavior consistent for the custom op path.
    if _use_custom_op:
        xyz1 = tf.expand_dims(xyz1, 0)
        xyz2 = tf.expand_dims(xyz2, 0)
        return _nn_mod.nn_distance(xyz1, xyz2)
    else:
        # CPU fallback expects proper batch dims already.
        # If your tensors are missing batch dim, uncomment next 2 lines.
        # xyz1 = tf.expand_dims(xyz1, 0)
        # xyz2 = tf.expand_dims(xyz2, 0)
        return _cpu_nn_distance(xyz1, xyz2)


# Gradient is only available for the compiled custom op
if _use_custom_op:
    @ops.RegisterGradient('NnDistance')
    def _nn_distance_grad(op, grad_dist1, grad_idx1, grad_dist2, grad_idx2):
        xyz1 = op.inputs[0]
        xyz2 = op.inputs[1]
        idx1 = op.outputs[1]
        idx2 = op.outputs[3]
        return _nn_mod.nn_distance_grad(xyz1, xyz2, grad_dist1, idx1, grad_dist2, idx2)
else:
    # If you attempt to train with CPU fallback, gradients may not be registered.
    # For Design A (demo/inference), this is fine.
    pass
