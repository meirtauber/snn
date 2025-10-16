#!/usr/bin/env python3
"""
Verification script to ensure the implementation is complete and working.
"""

import sys
from pathlib import Path

def check_file(path, description):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} NOT FOUND")
        return False

def check_dir(path, description):
    """Check if a directory exists."""
    if Path(path).exists() and Path(path).is_dir():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} NOT FOUND")
        return False

def main():
    print("\n" + "="*70)
    print("IMPLEMENTATION VERIFICATION")
    print("="*70 + "\n")
    
    all_checks = []
    
    # Check core files
    print("Core Implementation Files:")
    all_checks.append(check_file("models/dense_temporal_snn_depth.py", "Model architecture"))
    all_checks.append(check_file("utils/carla_diverse_dataset.py", "Dataset loader"))
    all_checks.append(check_file("scripts/train_dense_temporal.py", "Training script"))
    all_checks.append(check_file("utils/losses.py", "Loss functions"))
    all_checks.append(check_file("utils/metrics.py", "Metrics"))
    
    print("\nDocumentation:")
    all_checks.append(check_file("README_DENSE_TEMPORAL.md", "Main documentation"))
    all_checks.append(check_file("IMPLEMENTATION_SUMMARY.md", "Implementation summary"))
    
    print("\nDataset:")
    all_checks.append(check_dir("data/diverse_20k", "CARLA diverse dataset"))
    all_checks.append(check_file("data/diverse_20k/metadata.json", "Dataset metadata"))
    
    print("\nLegacy Files Removed:")
    legacy_removed = []
    legacy_removed.append(not Path("models/snn_depth.py").exists())
    legacy_removed.append(not Path("models/temporal_snn_depth.py").exists())
    legacy_removed.append(not Path("scripts/train.py").exists())
    legacy_removed.append(not Path("webcam").exists())
    legacy_removed.append(not Path("config.py").exists())
    
    if all(legacy_removed):
        print("✓ All legacy files removed")
        all_checks.append(True)
    else:
        print("✗ Some legacy files still present")
        all_checks.append(False)
    
    # Test imports
    print("\nTesting Imports:")
    try:
        from models.dense_temporal_snn_depth import DenseTemporalSNNDepth
        print("✓ Model imports successfully")
        all_checks.append(True)
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        all_checks.append(False)
    
    try:
        from utils.carla_diverse_dataset import CARLADiverseDataset
        print("✓ Dataset imports successfully")
        all_checks.append(True)
    except Exception as e:
        print(f"✗ Dataset import failed: {e}")
        all_checks.append(False)
    
    try:
        from utils.losses import CompositeLoss
        print("✓ Loss imports successfully")
        all_checks.append(True)
    except Exception as e:
        print(f"✗ Loss import failed: {e}")
        all_checks.append(False)
    
    # Summary
    print("\n" + "="*70)
    passed = sum(all_checks)
    total = len(all_checks)
    print(f"VERIFICATION RESULT: {passed}/{total} checks passed")
    
    if all(all_checks):
        print("✓ ALL CHECKS PASSED - Ready for training!")
    else:
        print("✗ SOME CHECKS FAILED - Please review above")
    print("="*70 + "\n")
    
    return 0 if all(all_checks) else 1

if __name__ == "__main__":
    sys.exit(main())
