import os
import json
import numpy as np
import requests
import torch
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List

# Define default channels
# Channel 0: Normalized distance matrix (1 / (1 + D))
# Channel 1: Normalized pLDDT interaction (pLDDT_i * pLDDT_j / 10000)
# Channel 2: Normalized PAE matrix (1 - PAE / 30)

def download_alphafold_data(uniprot_id: str, output_dir: str = "data") -> Tuple[Optional[str], Optional[str]]:
    """
    Downloads PDB and PAE files from the AlphaFold DB for a given UniProt ID.
    Queries the REST API to get the correct URL versions.
    """
    os.makedirs(output_dir, exist_ok=True)
    api_url = f"https://alphafold.com/api/prediction/{uniprot_id}"
    
    pdb_path = os.path.join(output_dir, f"AF-{uniprot_id}-F1-model.pdb")
    pae_path = os.path.join(output_dir, f"AF-{uniprot_id}-F1-predicted_aligned_error.json")
    
    # If already downloaded, return them
    if os.path.exists(pdb_path) and os.path.exists(pae_path):
        return pdb_path, pae_path

    try:
        response = requests.get(api_url, timeout=15)
        if response.status_code != 200:
            print(f"[-] UniProt ID {uniprot_id} not found in AlphaFold DB.")
            return None, None
            
        data = response.json()
        if not data or len(data) == 0:
            print(f"[-] Empty prediction details for {uniprot_id}.")
            return None, None
            
        pdb_url = data[0].get("pdbUrl")
        pae_url = data[0].get("paeImageUrl")  # Actually PAE json is linked in data, let's verify if paeUrl is present
        # Often PAE URL can be retrieved by swapping extension in PDB URL or via 'paeDocUrl' or 'paeImageUrl'
        # Let's inspect typical keys or construct it:
        # Example pdbUrl: https://alphafold.ebi.ac.uk/files/AF-P00520-F1-model_v4.pdb
        # Corresponding PAE: https://alphafold.ebi.ac.uk/files/AF-P00520-F1-predicted_aligned_error_v4.json
        if not pdb_url:
            print(f"[-] No PDB URL returned for {uniprot_id}.")
            return None, None
            
        # Construct PAE URL from PDB URL
        # e.g., AF-P00520-F1-model_v4.pdb -> AF-P00520-F1-predicted_aligned_error_v4.json
        if "model_v" in pdb_url:
            base_url = pdb_url.split("-model_v")[0]
            version = pdb_url.split("-model_v")[1].split(".pdb")[0]
            pae_url = f"{base_url}-predicted_aligned_error_v{version}.json"
        else:
            pae_url = pdb_url.replace("-model.pdb", "-predicted_aligned_error.json").replace(".pdb", ".json")

        print(f"[+] Downloading PDB for {uniprot_id}...")
        pdb_resp = requests.get(pdb_url, timeout=20)
        if pdb_resp.status_code == 200:
            with open(pdb_path, "w", encoding="utf-8") as f:
                f.write(pdb_resp.text)
        else:
            print(f"[-] Failed to download PDB from {pdb_url}")
            return None, None

        print(f"[+] Downloading PAE for {uniprot_id}...")
        pae_resp = requests.get(pae_url, timeout=20)
        if pae_resp.status_code == 200:
            with open(pae_path, "w", encoding="utf-8") as f:
                f.write(pae_resp.text)
        else:
            # If PAE JSON is not found, we can proceed without it (use zeros as fallback for PAE channel)
            print(f"[!] Warning: PAE file download failed from {pae_url}. Proceeding with PDB only.")
            pae_path = None

        return pdb_path, pae_path

    except Exception as e:
        print(f"[-] Error downloading data for {uniprot_id}: {e}")
        return None, None


def parse_pdb(pdb_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Parses a PDB file to extract C-alpha coordinates, pLDDT scores (B-factors), and amino acid residues.
    Returns:
        coords: numpy array of shape (N, 3)
        plddts: numpy array of shape (N,)
        sequence: list of residue names of length N
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    
    coords = []
    plddts = []
    sequence = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Make sure it's a standard amino acid residue (heteroatoms are excluded)
                if residue.id[0] != " ":
                    continue
                # We need the C-alpha atom (CA)
                if "CA" in residue:
                    ca_atom = residue["CA"]
                    coords.append(ca_atom.get_coord())
                    plddts.append(ca_atom.get_bfactor())  # B-factor holds pLDDT score in AlphaFold
                    sequence.append(residue.get_resname())
            # Break after first chain for simplicity (or parse all, AlphaFold predictions usually have one chain)
            break
        break
        
    return np.array(coords, dtype=np.float32), np.array(plddts, dtype=np.float32), sequence


def parse_pae(pae_path: str) -> np.ndarray:
    """
    Parses an AlphaFold PAE JSON file and returns the 2D PAE matrix.
    Returns a matrix of shape (N, N).
    """
    with open(pae_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # The JSON is typically a list containing a dict with "predicted_aligned_error" key
    if isinstance(data, list) and len(data) > 0:
        entry = data[0]
    elif isinstance(data, dict):
        entry = data
    else:
        raise ValueError("Invalid PAE JSON format.")
        
    if "predicted_aligned_error" in entry:
        return np.array(entry["predicted_aligned_error"], dtype=np.float32)
    elif "distance" in entry:
        # Older format had a flat list under "distance" and needs reshaping
        distances = np.array(entry["distance"], dtype=np.float32)
        n = int(np.sqrt(len(distances)))
        return distances.reshape(n, n)
    else:
        raise KeyError("Could not find predicted aligned error array in JSON.")


def generate_feature_matrix(pdb_path: str, pae_path: Optional[str] = None, size: int = 128) -> np.ndarray:
    """
    Generates a 3-channel 2D representation of shape (3, Size, Size) from PDB and PAE.
    Channel 0: Proximity Matrix = 1.0 / (1.0 + EuclideanDistance)
    Channel 1: pLDDT Interaction Matrix = pLDDT_i * pLDDT_j / 10000.0
    Channel 2: PAE Confidence Matrix = 1.0 - (PAE / 30.0) if PAE provided, else zeros.
    """
    coords, plddts, _ = parse_pdb(pdb_path)
    n_residues = len(coords)
    
    if n_residues == 0:
        raise ValueError(f"No C-alpha atoms found in PDB: {pdb_path}")
        
    # 1. Compute Euclidean Distance and convert to proximity
    dist_matrix = cdist(coords, coords, metric='euclidean')
    proximity_matrix = 1.0 / (1.0 + dist_matrix)
    
    # 2. Compute pLDDT interaction matrix
    plddt_norm = plddts / 100.0
    plddt_matrix = np.outer(plddt_norm, plddt_norm)
    
    # 3. Parse and normalize PAE matrix
    if pae_path and os.path.exists(pae_path):
        try:
            pae_matrix = parse_pae(pae_path)
            # Clip to max of 30, and normalize: 1.0 is highest confidence (0 error), 0.0 is lowest (30+ error)
            pae_matrix = np.clip(pae_matrix, 0.0, 30.0)
            pae_matrix_norm = 1.0 - (pae_matrix / 30.0)
            
            # Check length mismatch between PDB and PAE
            if pae_matrix_norm.shape[0] != n_residues:
                # Rescale or pad/truncate to match sequence length
                # For simplicity, truncate or pad to match n_residues
                pae_temp = np.zeros((n_residues, n_residues), dtype=np.float32)
                min_dim = min(n_residues, pae_matrix_norm.shape[0])
                pae_temp[:min_dim, :min_dim] = pae_matrix_norm[:min_dim, :min_dim]
                pae_matrix_norm = pae_temp
        except Exception as e:
            print(f"[!] Error parsing PAE, using zero-matrix: {e}")
            pae_matrix_norm = np.zeros((n_residues, n_residues), dtype=np.float32)
    else:
        pae_matrix_norm = np.zeros((n_residues, n_residues), dtype=np.float32)
        
    # Pack into (3, N, N)
    features = np.stack([proximity_matrix, plddt_matrix, pae_matrix_norm], axis=0)
    
    # Resize / pad / truncate to (3, Size, Size)
    c, h, w = features.shape
    padded = np.zeros((c, size, size), dtype=np.float32)
    
    # Crop if larger, or pad if smaller
    crop_h = min(h, size)
    crop_w = min(w, size)
    
    padded[:, :crop_h, :crop_w] = features[:, :crop_h, :crop_w]
    return padded


def generate_synthetic_pdb(output_path: str, n_residues: int = 150, plddt_mean: float = 85.0):
    """
    Generates a mock PDB file containing a 3D random walk of C-alpha atoms.
    Simulates high-quality, continuous coordinate chains.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Start coordinates
    x, y, z = 0.0, 0.0, 0.0
    
    residues = ["ALA", "GLY", "VAL", "LEU", "ILE", "PHE", "TYR", "TRP", "ASP", "GLU", 
                "LYS", "ARG", "HIS", "SER", "THR", "CYS", "MET", "ASN", "GLN", "PRO"]
                
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n_residues):
            # Step in random direction, average bond length between CA atoms is ~3.8 Angstroms
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            dx = 3.8 * np.sin(phi) * np.cos(theta)
            dy = 3.8 * np.sin(phi) * np.sin(theta)
            dz = 3.8 * np.cos(phi)
            
            x += dx
            y += dy
            z += dz
            
            # Simulate a realistic pLDDT score using a random walk bounded between 50 and 99
            plddt = np.clip(plddt_mean + np.random.normal(0, 5), 45.0, 99.0)
            res_name = residues[i % len(residues)]
            
            # Write a standard PDB ATOM record line
            # Formats: Atom name (CA), Residue name (res_name), chain ID (A), residue number (i+1),
            # coords (x,y,z), occupancy (1.00), B-factor/pLDDT (plddt)
            pdb_line = (
                f"ATOM  {i+1:5d}  CA  {res_name} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{plddt:6.2f}           C\n"
            )
            f.write(pdb_line)


def generate_synthetic_pae(output_path: str, n_residues: int = 150):
    """
    Generates a mock PAE JSON file with domain-like structures (block diagonal elements).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create block-diagonal structure (e.g. 2-3 domains)
    pae = np.random.uniform(15.0, 30.0, (n_residues, n_residues))
    
    # Make it symmetric-ish for simplicity and add domains
    n_domains = np.random.randint(2, 4)
    domain_boundaries = sorted([0] + list(np.random.choice(range(20, n_residues - 20), n_domains - 1, replace=False)) + [n_residues])
    
    for idx in range(len(domain_boundaries) - 1):
        start, end = domain_boundaries[idx], domain_boundaries[idx+1]
        # Domain internal PAE is low
        pae[start:end, start:end] = np.random.uniform(1.0, 8.0, (end-start, end-start))
        
    # Zero out diagonal (self-alignment has 0 error)
    np.fill_diagonal(pae, 0.0)
    
    # Save as JSON list
    data = [{
        "predicted_aligned_error": pae.tolist(),
        "max_predicted_aligned_error": 30.0
    }]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def generate_synthetic_dataset(output_dir: str = "data", num_samples: int = 50) -> List[Dict]:
    """
    Generates a synthetic set of PDB and PAE files, returning a list of dicts with filepaths and labels.
    Half of the proteins are simulated to be thermostable (label 1), half standard (label 0).
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_info = []
    
    print(f"[+] Generating {num_samples} synthetic protein structures...")
    
    for i in range(num_samples):
        # Even indices: Thermostable (label 1)
        # Odd indices: Standard (label 0)
        label = 1 if i % 2 == 0 else 0
        
        # Thermostable: shorter, more compact structure, higher pLDDT stability
        if label == 1:
            n_res = np.random.randint(80, 150)
            plddt_mean = 88.0
        else:
            n_res = np.random.randint(100, 220)
            plddt_mean = 72.0
            
        pdb_name = f"synthetic_mock_{i}.pdb"
        pae_name = f"synthetic_mock_{i}_pae.json"
        
        pdb_path = os.path.join(output_dir, pdb_name)
        pae_path = os.path.join(output_dir, pae_name)
        
        generate_synthetic_pdb(pdb_path, n_residues=n_res, plddt_mean=plddt_mean)
        generate_synthetic_pae(pae_path, n_residues=n_res)
        
        dataset_info.append({
            "pdb_path": pdb_path,
            "pae_path": pae_path,
            "label": label
        })
        
    return dataset_info


class AlphaFoldDataset(Dataset):
    """
    PyTorch Dataset that loads protein structure representations.
    Can run in real mode (downloads lists of UniProt IDs) or synthetic mode.
    """
    def __init__(self, data_list: List[Dict], size: int = 128):
        """
        Args:
            data_list: List of dicts, each with keys 'pdb_path', 'pae_path' (optional), and 'label'.
            size: Dimension of the output 2D representation.
        """
        self.data_list = data_list
        self.size = size
        
    def __len__(self) -> int:
        return len(self.data_list)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data_list[idx]
        pdb_path = item["pdb_path"]
        pae_path = item.get("pae_path")
        label = item["label"]
        
        features = generate_feature_matrix(pdb_path, pae_path, size=self.size)
        
        # Return features as torch Tensor, label as float (for binary cross entropy loss)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# Diagnostic script
if __name__ == "__main__":
    # Test synthetic generation
    print("[*] Running data_loader.py diagnosis...")
    samples = generate_synthetic_dataset("data/test_synthetic", num_samples=2)
    print(f"[+] Generated samples: {samples}")
    
    # Test feature compilation
    feat = generate_feature_matrix(samples[0]["pdb_path"], samples[0]["pae_path"], size=128)
    print(f"[+] Feature matrix shape: {feat.shape} (Expected: (3, 128, 128))")
    print(f"[+] Distance matrix (Ch 0) mean: {feat[0].mean():.4f}")
    print(f"[+] pLDDT matrix (Ch 1) mean: {feat[1].mean():.4f}")
    print(f"[+] PAE matrix (Ch 2) mean: {feat[2].mean():.4f}")
    
    # Plot feature channels for verification
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channels = ["Distance (Proximity)", "pLDDT Interaction", "PAE Confidence"]
    for idx, ax in enumerate(axes):
        im = ax.imshow(feat[idx], cmap="viridis", origin="lower")
        ax.set_title(channels[idx])
        fig.colorbar(im, ax=ax)
    
    plot_path = "data/test_synthetic/sample_features.png"
    plt.savefig(plot_path)
    print(f"[+] Diagnostic plots saved to {plot_path}")
