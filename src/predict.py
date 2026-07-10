import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional

from data_loader import generate_feature_matrix, parse_pdb
from model import AlphaFoldCNN


class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps
    to explain the predictions of the AlphaFoldCNN model.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register forward and backward hooks to capture feature maps and gradients
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate(self, target_score: torch.Tensor) -> np.ndarray:
        # 1. Zero out model gradients
        self.model.zero_grad()
        
        # 2. Backpropagate the target score (logit) to calculate gradients
        target_score.backward(retain_graph=True)
        
        # 3. Retrieve captured gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            raise RuntimeError("Gradients or activations were not captured. Verify hook registration.")
            
        # 4. Global Average Pooling (GAP) of gradients across spatial dimensions
        # Shape: [batch_size, channels, H, W] -> mean over batch, H, W -> [channels]
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # 5. Multiply each activation channel by its corresponding pooled gradient
        weighted_activations = activations * pooled_gradients.view(1, -1, 1, 1)
        
        # 6. Sum across channels to get the 2D saliency map
        heatmap = torch.sum(weighted_activations, dim=1).squeeze()
        
        # 7. Apply ReLU to keep only positive (stabilizing/destabilizing) contributions
        heatmap = torch.clamp(heatmap, min=0)
        
        # 8. Normalize to [0, 1]
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            
        # 9. Remove hooks to prevent memory leaks
        self.forward_hook.remove()
        self.backward_hook.remove()
        
        return heatmap.cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Protein Thermostability using AlphaFold-CNN")
    parser.add_argument("--pdb", type=str, required=True, help="Path to input PDB structure file")
    parser.add_argument("--pae", type=str, default=None, help="Path to input PAE JSON file (optional)")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to trained model weights")
    parser.add_argument("--size", type=int, default=128, help="Matrix dimension (Size x Size)")
    parser.add_argument("--plot", action="store_true", help="Plot and save 2D matrix representations")
    parser.add_argument("--output-plot", type=str, default="prediction_result.png", help="Path to save output visualization")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Verification of inputs
    if not os.path.exists(args.pdb):
        print(f"[-] Error: Input PDB file not found at: {args.pdb}")
        return
        
    if not os.path.exists(args.model):
        print(f"[-] Error: Trained model weights not found at: {args.model}")
        print("[-] Please run 'python src/train.py' first to train and save the model.")
        return
        
    # 2. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaFoldCNN(size=args.size).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
    except Exception as e:
        print(f"[-] Error loading model weights: {e}")
        return
        
    # 3. Generate Feature Tensor
    print(f"[+] Loading and parsing structure files...")
    try:
        features = generate_feature_matrix(args.pdb, args.pae, size=args.size)
    except Exception as e:
        print(f"[-] Error parsing structure: {e}")
        return
        
    # Create batch of 1
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Ensure input tensor requires grad to trace back to it
    input_tensor.requires_grad = True
    
    # Register Grad-CAM on conv4 (the last conv layer in AlphaFoldCNN)
    grad_cam = GradCAM(model, model.conv4)
    
    # 4. Model Inference (Gradients enabled for Grad-CAM)
    print(f"[+] Running neural network prediction...")
    logits = model(input_tensor)
    probability = torch.sigmoid(logits).detach().item()
        
    prediction_class = "Thermostable (High Stability)" if probability >= 0.5 else "Standard (Mesophilic / Normal Stability)"
    confidence = probability if probability >= 0.5 else (1.0 - probability)
    
    # 5. Print beautiful results
    print("\n" + "="*60)
    print(f"{'PREDICTION RESULT':^60}")
    print("="*60)
    print(f"  PDB File:    {os.path.basename(args.pdb)}")
    if args.pae:
        print(f"  PAE File:    {os.path.basename(args.pae)}")
    else:
        print(f"  PAE File:    Not provided (used zero-confidence fallback)")
    print(f"  Stability:   {prediction_class}")
    print(f"  Probability: {probability:.4f} ({probability:.2%})")
    print(f"  Confidence:  {confidence:.2%}")
    print("="*60 + "\n")
    
    # Generate Grad-CAM Map
    print(f"[+] Generating Grad-CAM saliency map...")
    # If prediction is Thermostable, target logit directly to find stabilizing regions
    # If prediction is Standard, target negative logit to find destabilizing regions
    target_score = logits if probability >= 0.5 else -logits
    
    heatmap_resized = None
    try:
        heatmap_raw = grad_cam.generate(target_score)
        
        # Upscale heatmap from conv4 output size (e.g., 16x16) to input size (e.g., 128x128)
        heatmap_tensor = torch.tensor(heatmap_raw, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heatmap_resized = F.interpolate(
            heatmap_tensor, size=(args.size, args.size), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        print(f"[+] Grad-CAM generated and upscaled successfully.")
    except Exception as e:
        print(f"[!] Warning: Grad-CAM generation failed: {e}")
    
    # 6. Plotting and Visualizations
    if args.plot:
        coords, plddts, seq = parse_pdb(args.pdb)
        seq_len = len(seq)
        
        # Setup a 4-panel visualization
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        
        # Plot 1: Proximity Matrix (Channel 0)
        im0 = axes[0].imshow(features[0], cmap="magma", origin="lower")
        axes[0].set_title(f"Backbone Proximity (1/(1+D))\n(Parsed Residues: {seq_len})")
        axes[0].set_xlabel("Residue Index")
        axes[0].set_ylabel("Residue Index")
        fig.colorbar(im0, ax=axes[0])
        
        # Plot 2: pLDDT Matrix (Channel 1)
        im1 = axes[1].imshow(features[1], cmap="viridis", origin="lower")
        axes[1].set_title(f"pLDDT Interaction (Confidence)\n(Mean pLDDT: {np.mean(plddts):.1f})")
        axes[1].set_xlabel("Residue Index")
        axes[1].set_ylabel("Residue Index")
        fig.colorbar(im1, ax=axes[1])
        
        # Plot 3: PAE Matrix (Channel 2)
        im2 = axes[2].imshow(features[2], cmap="plasma", origin="lower")
        status_pae = "Loaded" if args.pae else "Fallback (Zeros)"
        axes[2].set_title(f"PAE Confidence Matrix (1 - PAE/30)\n(PAE Data: {status_pae})")
        axes[2].set_xlabel("Residue Index")
        axes[2].set_ylabel("Residue Index")
        fig.colorbar(im2, ax=axes[2])
        
        # Plot 4: Grad-CAM Saliency Overlay
        # Show proximity matrix in grayscale as structural context
        axes[3].imshow(features[0], cmap="gray", origin="lower")
        if heatmap_resized is not None:
            # Overlay heatmap using semi-transparent jet colormap
            im3 = axes[3].imshow(heatmap_resized, cmap="jet", alpha=0.55, origin="lower")
            title_class = "Stabilizing Regions" if probability >= 0.5 else "Destabilizing Regions"
            axes[3].set_title(f"Grad-CAM Saliency ({title_class})\n(Target: {'+Logit' if probability >= 0.5 else '-Logit'})")
            fig.colorbar(im3, ax=axes[3])
        else:
            axes[3].set_title("Grad-CAM Saliency\n(Failed to generate)")
            
        axes[3].set_xlabel("Residue Index")
        axes[3].set_ylabel("Residue Index")
        
        fig.suptitle(f"Thermostability Prediction: {prediction_class} (Prob: {probability:.2%})", 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(args.output_plot, bbox_inches="tight")
        print(f"[+] Prediction visualization saved to {args.output_plot}")


if __name__ == "__main__":
    main()
