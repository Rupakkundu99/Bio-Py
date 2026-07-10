import os
import shutil
import pytest
import torch
import numpy as np

from src.data_loader import (
    generate_synthetic_pdb,
    generate_synthetic_pae,
    generate_synthetic_dataset,
    parse_pdb,
    parse_pae,
    generate_feature_matrix,
    AlphaFoldDataset
)
from src.model import AlphaFoldCNN

TEMP_TEST_DIR = "data/test_temp"

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup test directory
    os.makedirs(TEMP_TEST_DIR, exist_ok=True)
    yield
    # Cleanup after all tests in module finish
    if os.path.exists(TEMP_TEST_DIR):
        shutil.rmtree(TEMP_TEST_DIR)


def test_synthetic_pdb_generation():
    pdb_path = os.path.join(TEMP_TEST_DIR, "test_mock.pdb")
    n_res = 120
    generate_synthetic_pdb(pdb_path, n_residues=n_res, plddt_mean=80.0)
    
    assert os.path.exists(pdb_path)
    
    # Verify parsing
    coords, plddts, sequence = parse_pdb(pdb_path)
    assert coords.shape == (n_res, 3)
    assert plddts.shape == (n_res,)
    assert len(sequence) == n_res
    assert sequence[0] in ["ALA", "GLY", "VAL", "LEU", "ILE", "PHE", "TYR", "TRP", "ASP", "GLU", 
                           "LYS", "ARG", "HIS", "SER", "THR", "CYS", "MET", "ASN", "GLN", "PRO"]


def test_synthetic_pae_generation():
    pae_path = os.path.join(TEMP_TEST_DIR, "test_mock_pae.json")
    n_res = 120
    generate_synthetic_pae(pae_path, n_residues=n_res)
    
    assert os.path.exists(pae_path)
    
    # Verify parsing
    pae_matrix = parse_pae(pae_path)
    assert pae_matrix.shape == (n_res, n_res)
    assert np.all(pae_matrix >= 0.0)
    assert np.all(pae_matrix <= 30.0)
    assert pae_matrix[0, 0] == 0.0  # diagonal is 0


def test_feature_matrix_shapes():
    pdb_path = os.path.join(TEMP_TEST_DIR, "test_mock.pdb")
    pae_path = os.path.join(TEMP_TEST_DIR, "test_mock_pae.json")
    n_res = 100
    
    generate_synthetic_pdb(pdb_path, n_residues=n_res)
    generate_synthetic_pae(pae_path, n_residues=n_res)
    
    # Test generation with size=128
    feat_128 = generate_feature_matrix(pdb_path, pae_path, size=128)
    assert feat_128.shape == (3, 128, 128)
    
    # Test generation with size=64
    feat_64 = generate_feature_matrix(pdb_path, pae_path, size=64)
    assert feat_64.shape == (3, 64, 64)
    
    # Test without PAE file (fallback channel 2 is zero)
    feat_no_pae = generate_feature_matrix(pdb_path, pae_path=None, size=128)
    assert feat_no_pae.shape == (3, 128, 128)
    assert np.all(feat_no_pae[2] == 0.0)


def test_pytorch_dataset():
    # Generate a dataset list of size 4
    dataset_info = generate_synthetic_dataset(TEMP_TEST_DIR, num_samples=4)
    dataset = AlphaFoldDataset(dataset_info, size=128)
    
    assert len(dataset) == 4
    
    features, label = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert features.shape == (3, 128, 128)
    assert label.item() in [0.0, 1.0]


def test_cnn_model_forward():
    batch_size = 4
    size = 128
    dummy_input = torch.randn(batch_size, 3, size, size)
    
    model = AlphaFoldCNN(size=size)
    output = model(dummy_input)
    
    assert output.shape == (batch_size, 1)
    
    # Test with different size
    dummy_input_64 = torch.randn(batch_size, 3, 64, 64)
    # The architecture should be robust to size changes due to adaptive average pooling
    output_64 = model(dummy_input_64)
    assert output_64.shape == (batch_size, 1)
