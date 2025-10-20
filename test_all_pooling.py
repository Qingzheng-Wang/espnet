import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from espnet2.spk.pooling.chn_attn_stat_pooling import (
    ChnAttnStatPooling,
    ChnAttnStatPoolingOrigin,
)
from espnet2.spk.pooling.mean_pooling import (
    MeanPooling,
    MeanPoolingOrigin,
)
from espnet2.spk.pooling.stat_pooling import (
    StatsPooling,
    StatsPoolingOrigin,
)


def sync_model_parameters(model1, model2):
    """Synchronize parameters from model1 to model2 to ensure identical weights."""
    # Only sync if both models have parameters (e.g., ChnAttnStatPooling has attention)
    if list(model1.parameters()) and list(model2.parameters()):
        model2.load_state_dict(model1.state_dict())


def compare_tensors(tensor1, tensor2, name, rtol=1e-5, atol=1e-5):
    """Compare two tensors and report if they are close."""
    if torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        print(f"✓ {name}: PASSED (max diff: {torch.max(torch.abs(tensor1 - tensor2)).item():.2e})")
        return True
    else:
        max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
        mean_diff = torch.mean(torch.abs(tensor1 - tensor2)).item()
        print(f"✗ {name}: FAILED (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})")
        print(f"  tensor1 sample: {tensor1[0, :5]}")
        print(f"  tensor2 sample: {tensor2[0, :5]}")
        return False


def test_pooling(
    new_model_class,
    origin_model_class,
    batch_size,
    feature_dim,
    seq_len,
    feat_lengths=None,
    test_name="Test",
):
    """Test pooling with given parameters."""
    print(f"\n{'='*60}")
    print(f"{test_name}")
    print(f"  batch_size={batch_size}, feature_dim={feature_dim}, seq_len={seq_len}")
    if feat_lengths is not None:
        print(f"  feat_lengths={feat_lengths.tolist()}")
    print(f"{'='*60}")

    # Create models
    model_new = new_model_class(input_size=feature_dim)
    model_origin = origin_model_class(input_size=feature_dim)
    
    # Synchronize parameters to ensure identical weights (for models with parameters)
    sync_model_parameters(model_new, model_origin)
    
    # Set to evaluation mode
    model_new.eval()
    model_origin.eval()

    # Create random input
    torch.manual_seed(42)
    x = torch.randn(batch_size, feature_dim, seq_len)
    
    # Forward pass
    with torch.no_grad():
        output_new = model_new(x, feat_lengths)
        output_origin = model_origin(x, feat_lengths)
    
    # Compare outputs
    passed = compare_tensors(
        output_new,
        output_origin,
        f"{test_name} - Output",
        rtol=1e-4,
        atol=1e-5,
    )
    
    return passed


def run_pooling_tests(
    pooling_name,
    new_model_class,
    origin_model_class,
):
    """Run tests for a specific pooling type."""
    print("\n" + "="*70)
    print(f"TESTING {pooling_name}")
    print("="*70)
    
    test_count = 0
    passed_count = 0
    
    # Test 1: Basic test without feat_lengths
    test_count += 1
    if test_pooling(
        new_model_class,
        origin_model_class,
        batch_size=2,
        feature_dim=128,
        seq_len=50,
        feat_lengths=None,
        test_name=f"{pooling_name} - Test 1: Basic (no feat_lengths)",
    ):
        passed_count += 1
    
    # Test 2: With feat_lengths (no padding)
    test_count += 1
    if test_pooling(
        new_model_class,
        origin_model_class,
        batch_size=2,
        feature_dim=128,
        seq_len=50,
        feat_lengths=torch.tensor([50, 50]),
        test_name=f"{pooling_name} - Test 2: With feat_lengths (no padding)",
    ):
        passed_count += 1
    
    # Test 3: With feat_lengths (with padding)
    test_count += 1
    if test_pooling(
        new_model_class,
        origin_model_class,
        batch_size=3,
        feature_dim=128,
        seq_len=100,
        feat_lengths=torch.tensor([80, 60, 100]),
        test_name=f"{pooling_name} - Test 3: With feat_lengths (with padding)",
    ):
        passed_count += 1
    
    # Test 4: Large batch size
    test_count += 1
    if test_pooling(
        new_model_class,
        origin_model_class,
        batch_size=8,
        feature_dim=256,
        seq_len=75,
        feat_lengths=torch.tensor([75, 70, 65, 60, 55, 50, 45, 75]),
        test_name=f"{pooling_name} - Test 4: Large batch size",
    ):
        passed_count += 1
    
    # Test 5: Different feature dimensions
    test_count += 1
    if test_pooling(
        new_model_class,
        origin_model_class,
        batch_size=4,
        feature_dim=512,
        seq_len=30,
        feat_lengths=torch.tensor([30, 25, 20, 15]),
        test_name=f"{pooling_name} - Test 5: Different feature dimensions (512)",
    ):
        passed_count += 1
    
    # Test 6: Very short sequences
    test_count += 1
    if test_pooling(
        new_model_class,
        origin_model_class,
        batch_size=2,
        feature_dim=128,
        seq_len=10,
        feat_lengths=torch.tensor([8, 10]),
        test_name=f"{pooling_name} - Test 6: Very short sequences",
    ):
        passed_count += 1
    
    # Test 7: Single sample
    test_count += 1
    if test_pooling(
        new_model_class,
        origin_model_class,
        batch_size=1,
        feature_dim=128,
        seq_len=50,
        feat_lengths=torch.tensor([40]),
        test_name=f"{pooling_name} - Test 7: Single sample",
    ):
        passed_count += 1
    
    # Test 8: Default feature dimension (1536)
    test_count += 1
    if test_pooling(
        new_model_class,
        origin_model_class,
        batch_size=2,
        feature_dim=1536,
        seq_len=50,
        feat_lengths=torch.tensor([45, 50]),
        test_name=f"{pooling_name} - Test 8: Default feature dimension (1536)",
    ):
        passed_count += 1
    
    # Test 9: Very small feat_lengths
    test_count += 1
    if test_pooling(
        new_model_class,
        origin_model_class,
        batch_size=3,
        feature_dim=128,
        seq_len=100,
        feat_lengths=torch.tensor([5, 10, 15]),
        test_name=f"{pooling_name} - Test 9: Very small feat_lengths",
    ):
        passed_count += 1
    
    # Test 10: Edge case - feat_length = 1
    test_count += 1
    if test_pooling(
        new_model_class,
        origin_model_class,
        batch_size=2,
        feature_dim=128,
        seq_len=50,
        feat_lengths=torch.tensor([1, 2]),
        test_name=f"{pooling_name} - Test 10: Edge case (feat_length = 1)",
    ):
        passed_count += 1
    
    print(f"\n{pooling_name} Summary: {passed_count}/{test_count} tests passed")
    
    return test_count, passed_count


def run_all_tests():
    """Run comprehensive tests for all pooling types."""
    print("\n" + "="*70)
    print("COMPREHENSIVE POOLING COMPARISON TEST SUITE")
    print("="*70)
    
    total_tests = 0
    total_passed = 0
    
    pooling_configs = [
        ("MeanPooling", MeanPooling, MeanPoolingOrigin),
        ("StatsPooling", StatsPooling, StatsPoolingOrigin),
        ("ChnAttnStatPooling", ChnAttnStatPooling, ChnAttnStatPoolingOrigin),
    ]
    
    results = {}
    
    for pooling_name, new_class, origin_class in pooling_configs:
        test_count, passed_count = run_pooling_tests(pooling_name, new_class, origin_class)
        total_tests += test_count
        total_passed += passed_count
        results[pooling_name] = (passed_count, test_count)
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    for pooling_name, (passed, total) in results.items():
        status = "✓ PASS" if passed == total else f"✗ FAIL ({total - passed} failures)"
        print(f"{pooling_name:25s}: {passed:2d}/{total:2d} tests {status}")
    
    print(f"\n{'='*70}")
    print(f"Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ SOME TESTS FAILED ({total_tests - total_passed} failures)")
    
    print("="*70)
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

