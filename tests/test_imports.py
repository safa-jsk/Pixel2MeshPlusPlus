#!/usr/bin/env python3
"""
Smoke test: verify the refactored package structure is importable.

Run:  python tests/test_imports.py
"""
import os
import sys

# Bootstrap project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

PASS = 0
FAIL = 0


def check(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


def main():
    print("=" * 60)
    print("Smoke Test: Package Structure & Imports")
    print("=" * 60)

    # 1. Check directory structure exists
    print("\n--- Directory structure ---")
    dirs = [
        "src/p2mpp",
        "src/p2mpp/tf",
        "src/p2mpp/tf/modules",
        "src/p2mpp/tf/utils",
        "src/p2mpp/tf/scripts",
        "src/p2mpp/torch",
        "src/p2mpp/torch/modules",
        "src/p2mpp/torch/engine",
        "src/p2mpp/torch/convert",
        "src/p2mpp/torch/utils",
        "DesignA_CPU/scripts",
        "DesignA_GPU/scripts",
        "DesignB/scripts",
        "DesignC/scripts",
        "external/tf_ops",
        "external/torch_chamfer",
        "configs/designA",
        "configs/designB",
        "assets/data_templates",
        "assets/demo_inputs",
        "docs",
        "tests",
    ]
    for d in dirs:
        check(d, os.path.isdir(os.path.join(PROJECT_ROOT, d)))

    # 2. Check key files exist
    print("\n--- Key files ---")
    files = [
        "src/p2mpp/__init__.py",
        "src/p2mpp/tf/__init__.py",
        "src/p2mpp/tf/_bootstrap.py",
        "src/p2mpp/tf/modules/config.py",
        "src/p2mpp/tf/modules/layers.py",
        "src/p2mpp/tf/modules/losses.py",
        "src/p2mpp/tf/modules/models_mvp2m.py",
        "src/p2mpp/tf/modules/models_p2mpp.py",
        "src/p2mpp/tf/modules/chamfer.py",
        "src/p2mpp/tf/utils/dataloader.py",
        "src/p2mpp/tf/utils/tools.py",
        "src/p2mpp/torch/modules/models_mvp2m_pytorch.py",
        "src/p2mpp/torch/modules/models_p2mpp_exact.py",
        "src/p2mpp/torch/engine/fast_inference_v4.py",
        "src/p2mpp/torch/engine/fast_inference_v4_metrics.py",
        "configs/designA/mvp2m.yaml",
        "configs/designA/p2mpp.yaml",
        "configs/designB/p2mpp_pytorch.yaml",
        "assets/data_templates/iccv_p2mpp.dat",
        "assets/data_templates/face3.obj",
        ".gitignore",
        "docs/index.md",
    ]
    for f in files:
        check(f, os.path.isfile(os.path.join(PROJECT_ROOT, f)))

    # 3. Check __init__.py has content
    print("\n--- Package init files ---")
    init_path = os.path.join(PROJECT_ROOT, "src", "p2mpp", "__init__.py")
    check("src/p2mpp/__init__.py is non-empty or exists",
          os.path.isfile(init_path))

    # 4. Check .gitignore contains artifacts
    print("\n--- .gitignore ---")
    with open(os.path.join(PROJECT_ROOT, ".gitignore"), "r") as f:
        gitignore = f.read()
    check("artifacts/** in .gitignore", "artifacts/**" in gitignore)
    check("external/torch_chamfer/build/ in .gitignore",
          "external/torch_chamfer/build/" in gitignore)

    # 5. Try importing p2mpp package
    print("\n--- Python imports ---")
    try:
        import src.p2mpp
        check("import src.p2mpp", True)
    except Exception as e:
        check(f"import src.p2mpp (error: {e})", False)

    # 6. Check DesignC stubs
    print("\n--- DesignC stubs ---")
    check("DesignC/scripts/infer_facescape.py",
          os.path.isfile(os.path.join(PROJECT_ROOT, "DesignC/scripts/infer_facescape.py")))
    check("DesignC/scripts/eval_facescape.py",
          os.path.isfile(os.path.join(PROJECT_ROOT, "DesignC/scripts/eval_facescape.py")))

    # Summary
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"Results: {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"WARNING: {FAIL} check(s) failed")
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
