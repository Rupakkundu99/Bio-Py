import os
import sys
import io
import base64
import requests
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gradio as gr
from typing import Tuple, Optional, List

# Add 'src' directory to Python path to support internal imports inside src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Load environment variables from .env file if it exists
def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

load_env_file()

# Try to import spaces for Hugging Face ZeroGPU compatibility
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

if not has_spaces:
    class MockSpaces:
        def GPU(self, fn=None, duration=None):
            if fn is not None:
                return fn
            def decorator(f):
                return f
            return decorator
    spaces = MockSpaces()

# Dummy function to satisfy Hugging Face ZeroGPU startup check.
# This is never called but must exist so Hugging Face allows the container to boot.
@spaces.GPU
def dummy_gpu_trigger():
    pass

# Import local pipeline utilities
from data_loader import generate_feature_matrix, parse_pdb, download_alphafold_data
from model import AlphaFoldCNN
from predict import GradCAM

def analyze_structure_heuristics(
    pdb_path: str,
    pae_path: Optional[str],
    features: np.ndarray,
    heatmap_resized: Optional[np.ndarray]
) -> Tuple[float, str, List[str], List[str], Optional[float], float]:
    """
    Computes local biophysical properties of the protein structure and matches
    them against the model's Grad-CAM heatmap.
    """
    coords, plddts, seq = parse_pdb(pdb_path)
    seq_len = len(seq)
    
    # 1. Compactness (mean of backbone proximity matrix)
    mean_proximity = float(np.mean(features[0]))
    if mean_proximity > 0.35:
        compactness_rating = "High"
    elif mean_proximity > 0.2:
        compactness_rating = "Moderate"
    else:
        compactness_rating = "Low"
        
    # 2. Low-Confidence Regions (pLDDT < 70)
    low_conf_residues = []
    for i, plddt in enumerate(plddts):
        if plddt < 70:
            low_conf_residues.append(i + 1)
            
    # Group contiguous low-confidence residues
    low_conf_ranges = []
    if low_conf_residues:
        start = low_conf_residues[0]
        prev = low_conf_residues[0]
        for r in low_conf_residues[1:]:
            if r == prev + 1:
                prev = r
            else:
                if start == prev:
                    low_conf_ranges.append(str(start))
                else:
                    low_conf_ranges.append(f"{start}-{prev}")
                start = r
                prev = r
        if start == prev:
            low_conf_ranges.append(str(start))
        else:
            low_conf_ranges.append(f"{start}-{prev}")
            
    # 3. Grad-CAM hotspots (where normalized 1D importance > 0.6)
    hotspots = []
    if heatmap_resized is not None:
        # Sum columns to get 1D residue contribution from the 2D interaction map
        residue_importance = np.mean(heatmap_resized, axis=0)
        max_imp = np.max(residue_importance)
        if max_imp > 0:
            residue_importance /= max_imp
            
        important_res = [i + 1 for i, val in enumerate(residue_importance) if val > 0.6 and i < seq_len]
        
        # Group contiguous hotspot residues
        if important_res:
            start = important_res[0]
            prev = important_res[0]
            for r in important_res[1:]:
                if r == prev + 1:
                    prev = r
                else:
                    if start == prev:
                        hotspots.append(str(start))
                    else:
                        hotspots.append(f"{start}-{prev}")
                    start = r
                    prev = r
            if start == prev:
                hotspots.append(str(start))
            else:
                hotspots.append(f"{start}-{prev}")
                
    # 4. PAE alignment statistics
    mean_pae_conf = None
    if pae_path and os.path.exists(pae_path):
        mean_pae_conf = float(np.mean(features[2]))
        
    return mean_proximity, compactness_rating, low_conf_ranges, hotspots, mean_pae_conf, float(np.mean(plddts))

def format_heuristic_report(
    mean_proximity: float,
    compactness_rating: str,
    low_conf_ranges: List[str],
    hotspots: List[str],
    mean_pae_conf: Optional[float],
    mean_plddt: float,
    prediction_class: str
) -> str:
    """
    Formats the heuristic findings into a clean, professional markdown report without emojis.
    """
    role = "stabilizing" if "Thermostable" in prediction_class else "destabilizing"
    
    report = "### Biophysical Feature Report\n\n"
    report += f"* **Compactness Index**: {mean_proximity:.4f} ({compactness_rating} folding compactness)\n"
    report += f"* **Mean structural confidence (pLDDT)**: {mean_plddt:.1f}/100\n"
    
    if mean_pae_conf is not None:
        report += f"* **Mean alignment confidence (PAE)**: {mean_pae_conf:.4f} (scale 0-1, where 1 represents zero error)\n"
    else:
        report += "* **Alignment confidence (PAE)**: PAE data not provided, using flat fallback matrix\n"
        
    if low_conf_ranges:
        report += f"* **Low-confidence structural regions (pLDDT < 70)**: Residues {', '.join(low_conf_ranges)}\n"
        report += "  These flexible or disordered regions represent primary candidates for stabilization (e.g., through proline substitution, salt bridge introduction, or disulfide bond design).\n"
    else:
        report += "* **Low-confidence structural regions**: None detected. The protein exhibits a highly stable structural core.\n"
        
    if hotspots:
        report += f"* **Model attention hotspots (Grad-CAM)**: Residues {', '.join(hotspots)}\n"
        report += f"  These residue interactions contributed most heavily to the network classifying the protein as {role}. Focus engineering efforts on these hotspot zones to modify stability.\n"
    else:
        report += "* **Model attention hotspots**: No individual residues exceeded the attention threshold.\n"
        
    return report

def predict_protein(
    pdb_file: Optional[object], 
    pae_file: Optional[object], 
    uniprot_id: str, 
    size: int, 
    model_path: str
) -> Tuple[Optional[plt.Figure], str, str, Optional[str]]:
    """
    Processes inputs, runs model inference, generates Grad-CAM plots, calculates 
    local biophysical heuristics, and saves a temporary copy of the plot for LLM access.
    """
    pdb_path = None
    pae_path = None
    
    # 1. Resolve inputs
    uniprot_id_clean = uniprot_id.strip()
    if uniprot_id_clean:
        download_dir = "data/downloads"
        os.makedirs(download_dir, exist_ok=True)
        try:
            pdb_path, pae_path = download_alphafold_data(uniprot_id_clean, output_dir=download_dir)
            if not pdb_path or not os.path.exists(pdb_path):
                return None, f"### Error\nCould not fetch PDB data for UniProt ID: **{uniprot_id_clean}** from AlphaFold DB.", "", None
        except Exception as e:
            return None, f"### Error\nFailed downloading UniProt ID **{uniprot_id_clean}**: {str(e)}", "", None
    else:
        if pdb_file is None:
            return None, "### Warning\nPlease upload a PDB file OR enter a UniProt ID.", "", None
        pdb_path = pdb_file.name
        pae_path = pae_file.name if pae_file else None

    # 2. Check model weights
    if not os.path.exists(model_path):
        return None, f"### Error\nTrained model weights not found at: `{model_path}`\n\nPlease train the model first by running `python src/train.py`.", "", None

    # 3. Setup device & load model
    # Force CPU on Hugging Face to avoid ZeroGPU quota checks for users
    is_hf = "SPACE_ID" in os.environ
    device = torch.device("cpu") if is_hf else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaFoldCNN(size=size).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        return None, f"### Error\nFailed loading model weights: {str(e)}", "", None

    # 4. Generate features
    try:
        features = generate_feature_matrix(pdb_path, pae_path, size=size)
    except Exception as e:
        return None, f"### Error\nParsing structure failed: {str(e)}", "", None

    # 5. Run inference with Grad-CAM active
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    input_tensor.requires_grad = True
    
    grad_cam = GradCAM(model, model.conv4)
    logits = model(input_tensor)
    probability = torch.sigmoid(logits).detach().item()
    
    prediction_class = "Thermostable" if probability >= 0.5 else "Standard"
    confidence = probability if probability >= 0.5 else (1.0 - probability)
    
    # 6. Generate Grad-CAM saliency map
    target_score = logits if probability >= 0.5 else -logits
    heatmap_resized = None
    try:
        heatmap_raw = grad_cam.generate(target_score)
        heatmap_tensor = torch.tensor(heatmap_raw, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heatmap_resized = F.interpolate(
            heatmap_tensor, size=(size, size), mode="bilinear", align_corners=False
        ).squeeze().numpy()
    except Exception as e:
        print(f"[!] Grad-CAM failed: {e}")

    # 7. Render 4-panel visualization
    try:
        coords, plddts, seq = parse_pdb(pdb_path)
        seq_len = len(seq)
        
        plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Plot 0: Backbone Proximity (Channel 0)
        im0 = axes[0].imshow(features[0], cmap="magma", origin="lower")
        axes[0].set_title(f"Backbone Proximity\n(Parsed Residues: {seq_len})", fontsize=10, fontweight="bold")
        axes[0].set_xlabel("Residue Index", fontsize=8)
        axes[0].set_ylabel("Residue Index", fontsize=8)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Plot 1: pLDDT Interaction (Channel 1)
        im1 = axes[1].imshow(features[1], cmap="viridis", origin="lower")
        axes[1].set_title(f"pLDDT Interaction (Confidence)\n(Mean pLDDT: {np.mean(plddts):.1f})", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("Residue Index", fontsize=8)
        axes[1].set_ylabel("Residue Index", fontsize=8)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Plot 2: PAE Confidence (Channel 2)
        im2 = axes[2].imshow(features[2], cmap="plasma", origin="lower")
        status_pae = "Loaded" if pae_path else "Fallback (Zeros)"
        axes[2].set_title(f"PAE Confidence\n(PAE Data: {status_pae})", fontsize=10, fontweight="bold")
        axes[2].set_xlabel("Residue Index", fontsize=8)
        axes[2].set_ylabel("Residue Index", fontsize=8)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Plot 3: Grad-CAM Saliency Overlay
        axes[3].imshow(features[0], cmap="gray", origin="lower")
        if heatmap_resized is not None:
            im3 = axes[3].imshow(heatmap_resized, cmap="jet", alpha=0.55, origin="lower")
            title_class = "Stabilizing Regions" if probability >= 0.5 else "Destabilizing Regions"
            axes[3].set_title(f"Grad-CAM Saliency\n({title_class})", fontsize=10, fontweight="bold")
            fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        else:
            axes[3].set_title("Grad-CAM Saliency\n(Failed to generate)", fontsize=10, fontweight="bold")
        axes[3].set_xlabel("Residue Index", fontsize=8)
        axes[3].set_ylabel("Residue Index", fontsize=8)
        
        fig.suptitle(
            f"Thermostability Prediction: {prediction_class} (Probability: {probability:.2%})", 
            fontsize=13, y=0.98, fontweight="bold"
        )
        plt.tight_layout()
        
        # Save temporary image file for Gemini access
        temp_dir = "data/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_plot_path = os.path.join(temp_dir, "latest_prediction_plot.png")
        fig.savefig(temp_plot_path, format="png", bbox_inches="tight", dpi=150)
        
        # 8. Compute Heuristics
        h_prox, h_comp, h_low, h_hotspots, h_pae, h_plddt = analyze_structure_heuristics(
            pdb_path, pae_path, features, heatmap_resized
        )
        
        heuristic_report = format_heuristic_report(
            h_prox, h_comp, h_low, h_hotspots, h_pae, h_plddt, prediction_class
        )
        
        # 9. Format HTML Summary Table
        status_color = "#10B981" if probability >= 0.5 else "#EF4444"
        summary_html = f"""
        <div style="background-color: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 6px solid {status_color}; font-family: sans-serif; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #111827;">Prediction Results</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 8px 0; font-weight: bold; color: #4B5563;">PDB Source:</td>
                    <td style="padding: 8px 0; font-family: monospace; color: #111827;">{os.path.basename(pdb_path)}</td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 8px 0; font-weight: bold; color: #4B5563;">Stability Outcome:</td>
                    <td style="padding: 8px 0; font-weight: bold; color: {status_color}; font-size: 1.1em;">{prediction_class.upper()}</td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 8px 0; font-weight: bold; color: #4B5563;">Thermostability Probability:</td>
                    <td style="padding: 8px 0; font-family: monospace; color: #111827; font-weight: bold;">{probability:.4f} ({probability:.2%})</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: bold; color: #4B5563;">Model Confidence:</td>
                    <td style="padding: 8px 0; font-family: monospace; color: #111827; font-weight: bold;">{confidence:.2%}</td>
                </tr>
            </table>
        </div>
        """
        
        return fig, summary_html, heuristic_report, temp_plot_path
        
    except Exception as e:
        return None, f"### Error\nVisualization rendering failed: {str(e)}", "", None

def generate_ai_explanation(
    plot_path: Optional[str],
    api_key: str,
    model_name: str,
    summary_html: str,
    heuristic_report: str
) -> str:
    """
    Sends the cached 4-panel image and analysis parameters to the Gemini API to get 
    a professional scientific explanation of the structural stability findings.
    """
    if not plot_path or not os.path.exists(plot_path):
        return "Please run a protein thermostability prediction first."
        
    if not api_key or not api_key.strip():
        return "Configure your Google Gemini API Key in the API Configuration panel to generate an AI explanation."
        
    api_key = api_key.strip()
    model_name = model_name.strip() if model_name else "gemini-2.5-flash"
    
    # 1. Base64 encode the saved figure image
    try:
        with open(plot_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return f"Failed to read visualization plot file: {str(e)}"
        
    # 2. Query Gemini API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    prompt = f"""You are a computational biophysicist analyzing structural features for protein engineering.
A 2D Convolutional Neural Network (CNN) has predicted the thermostability of a protein.

Model Prediction Context:
{summary_html}

Local Biophysical Heuristic Analysis:
{heuristic_report}

Analyze the attached 4-panel plot:
1. Backbone Proximity (Channel 0)
2. pLDDT Interaction (Channel 1)
3. PAE Confidence (Channel 2)
4. Grad-CAM Saliency Overlay (Channel 3)

Provide a structural analysis of the protein's properties:
- Discuss the relationship between proximity compactness, confidence, and predicted stability.
- Interpret the Grad-CAM saliency hotspots. Explain why the CNN likely focused on these residues and their biological significance.
- Suggest concrete protein engineering strategies (e.g., residues to mutate, loop stabilization, salt bridges) based on the low-confidence regions and Grad-CAM hotspots.

Do not use random emojis. Keep the formatting professional, scientific, and clean. Use markdown for structure."""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": img_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1200
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=40)
        if response.status_code != 200:
            return f"Gemini API Error (Status {response.status_code}): {response.text}"
            
        resp_data = response.json()
        candidates = resp_data.get("candidates", [])
        if not candidates:
            return "No candidates returned. The model might have filtered the content or the request was invalid."
            
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return "Empty content returned from the API."
            
        return parts[0].get("text", "")
        
    except Exception as e:
        return f"Network or API communication failure: {str(e)}"

# Custom CSS for modern design without random emojis
custom_css = """
footer {visibility: hidden}
#title-header {
    text-align: center;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 25px;
}
#title-header h1 {
    margin: 0;
    font-size: 2.2em;
    font-weight: 800;
}
#title-header p {
    margin: 5px 0 0 0;
    opacity: 0.9;
    font-size: 1.1em;
}
"""

with gr.Blocks() as demo:
    gr.HTML(
        """
        <div id="title-header">
            <h1>AF-Thermostability-CNN</h1>
            <p>Protein Thermostability Predictor using 2D CNN representations of AlphaFold predictions</p>
        </div>
        """
    )
    
    # State variable to pass latest generated plot filepath to the AI function
    plot_path_state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Inputs")
            
            with gr.Tabs():
                with gr.TabItem("Upload Files"):
                    pdb_input = gr.File(label="Upload Protein PDB (.pdb)", file_types=[".pdb"])
                    pae_input = gr.File(label="Upload PAE JSON (.json) [Optional]", file_types=[".json"])
                    
                with gr.TabItem("Fetch from AlphaFold DB"):
                    uniprot_input = gr.Textbox(
                        label="UniProt ID (e.g. P00520)", 
                        placeholder="Enter ID to download PDB and PAE automatically...",
                        value=""
                    )
            
            with gr.Accordion("Advanced Settings", open=False):
                size_input = gr.Slider(
                    label="Matrix Representation Size", 
                    minimum=64, 
                    maximum=256, 
                    step=16, 
                    value=128
                )
                model_input = gr.Textbox(
                    label="Model Weight Path", 
                    value="best_model.pth"
                )
                
            with gr.Accordion("API Configuration", open=False):
                api_key_input = gr.Textbox(
                    label="Google Gemini API Key", 
                    placeholder="Enter your API Key...",
                    type="password",
                    value=os.environ.get("GEMINI_API_KEY", "")
                )
                model_name_input = gr.Dropdown(
                    label="Gemini Model",
                    choices=[
                        "gemini-2.5-flash", 
                        "gemini-2.5-pro", 
                        "gemini-2.0-flash", 
                        "gemini-1.5-flash"
                    ],
                    value="gemini-2.5-flash",
                    allow_custom_value=True
                )
                
            run_btn = gr.Button("Predict Thermostability", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### Predictions and Explanations")
            status_output = gr.HTML(value="<div style='color: #6B7280; font-style: italic;'>Run prediction to see results...</div>")
            
            with gr.Tabs():
                with gr.TabItem("Visualizations"):
                    plot_output = gr.Plot(label="2D Channel Representations and Grad-CAM Hotspots")
                with gr.TabItem("Heuristic Report"):
                    heuristic_output = gr.Markdown(value="Local structural analysis will appear here...")
                with gr.TabItem("AI Biophysics Explanation"):
                    ai_btn = gr.Button("Generate AI Biophysics Explanation", variant="secondary")
                    ai_output = gr.Markdown(value="AI analysis will appear here after clicking the button above...")
            
    # Connect prediction trigger
    run_btn.click(
        fn=predict_protein,
        inputs=[pdb_input, pae_input, uniprot_input, size_input, model_input],
        outputs=[plot_output, status_output, heuristic_output, plot_path_state]
    )
    
    # Connect AI explanation trigger
    ai_btn.click(
        fn=generate_ai_explanation,
        inputs=[plot_path_state, api_key_input, model_name_input, status_output, heuristic_output],
        outputs=ai_output
    )
    
    # Example structures
    gr.Markdown("### Examples (Synthetic Mock Data)")
    gr.Examples(
        examples=[
            ["data/synthetic/synthetic_mock_0.pdb", "data/synthetic/synthetic_mock_0_pae.json", "", 128, "best_model.pth"],
            ["data/synthetic/synthetic_mock_1.pdb", "data/synthetic/synthetic_mock_1_pae.json", "", 128, "best_model.pth"],
            ["data/synthetic/synthetic_mock_2.pdb", None, "", 128, "best_model.pth"],
        ],
        inputs=[pdb_input, pae_input, uniprot_input, size_input, model_input],
        outputs=[plot_output, status_output, heuristic_output, plot_path_state],
        fn=predict_protein,
        cache_examples=False
    )

if __name__ == "__main__":
    # Hugging Face Spaces require the server to bind to 0.0.0.0 and listen on port 7860.
    # Locally, we check if port 7860 is occupied; if so, we fall back to port 7861.
    is_hf = "SPACE_ID" in os.environ
    server_name = "0.0.0.0" if is_hf else "127.0.0.1"
    server_port = 7860
    
    if not is_hf:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 7860))
            s.close()
        except OSError:
            server_port = 7861
            
    demo.launch(
        server_name=server_name, 
        server_port=server_port, 
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"), 
        css=custom_css
    )
