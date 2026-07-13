---
title: AF Thermostability CNN
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
---

# AF-Thermostability-CNN: Protein Thermostability Predictor


A production-ready machine learning pipeline that uses Google AlphaFold structural predictions to predict protein thermostability. The pipeline parses PDB and PAE structures, encodes them into a 3-channel 2D spatial representation (representing distance topology, local structural confidence, and alignment confidence), and processes them with a PyTorch 2D Convolutional Neural Network (CNN).

---

## 1. Key Concepts & Architecture
The project translates a 3D protein structure into a 2D spatial representation similar to an RGB image:
1.  **Channel 0 (Proximity)**: Bounded pairwise C-alpha distance map: $1.0 / (1.0 + \text{Distance})$.
2.  **Channel 1 (pLDDT Confidence)**: Outer product of residue-level confidence scores normalized: $\text{pLDDT}_i \times \text{pLDDT}_j / 10000.0$.
3.  **Channel 2 (PAE Accuracy)**: Alignment confidence derived from the Predicted Aligned Error: $1.0 - (\text{PAE} / 30.0)$.

The model is a 2D CNN with 4 convolutional blocks followed by a Global Average Pooling layer and a linear classification head.

---

## 2. Installation & Setup

First, clone or open the project folder. Ensure you have Python 3.8+ installed.

1.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    ```

2.  **Activate the Virtual Environment**:
    *   **Windows (PowerShell)**:
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *   **Windows (CMD)**:
        ```cmd
        .\venv\Scripts\activate.bat
        ```
    *   **macOS / Linux**:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 3. Running the Pipeline

### A. Run Unit Tests
To verify all parser and model shapes are correct, run the pytest suite:
```bash
pytest tests/
```

### B. Train the Model (Synthetic Fallback Mode)
The pipeline features a synthetic fallback generator that simulates PDB structures and PAE JSON matrices, allowing you to run and train the pipeline immediately without downloading large proteome datasets:
```bash
python src/train.py --epochs 10 --num-samples 50 --lr 0.001
```
This script will:
*   Generate 50 synthetic structures in `data/synthetic/` (25 thermostable, 25 standard).
*   Train the PyTorch CNN and output validation metrics (Accuracy, Precision, Recall, F1).
*   Save the best model checkpoints to `best_model.pth`.
*   Plot training loss/accuracy curves to `data/training_curves.png`.

### C. Predict on a New Structure
Use the inference tool to predict thermostability for a single structure.

1.  **Prediction without PAE file** (uses zeros for Channel 2):
    ```bash
    python src/predict.py --pdb data/synthetic/synthetic_mock_0.pdb --model best_model.pth --plot
    ```
2.  **Prediction with PAE file** (provides full 3-channel input):
    ```bash
    python src/predict.py --pdb data/synthetic/synthetic_mock_0.pdb --pae data/synthetic/synthetic_mock_0_pae.json --model best_model.pth --plot
    ```

The `--plot` flag generates a high-quality visualization named `prediction_result.png` containing plots for the 3 input channels along with the prediction outcome.

---

## 4. File Structure
*   `app.py`: Gradio web application for interactive predictions.
*   `src/data_loader.py`: Handles downloading from AlphaFold DB, parsing coordinates and B-factors, parsing PAE files, generating multi-channel matrices, padding/truncating, and synthetic data generation.
*   `src/model.py`: Neural network definition (4x Conv2D, MaxPool, AdaptiveAvgPool, Linear classification head).
*   `src/train.py`: Data loaders initialization, optimization, evaluation metric calculations, and training loop.
*   `src/predict.py`: Inference runner with plotting and logging tools.
*   `tests/test_pipeline.py`: Comprehensive unit tests.

---

## 5. Hugging Face Spaces Deployment

You can deploy this interactive interface directly on Hugging Face Spaces in just a few steps:

### A. Create a New Hugging Face Space
1. Log in to [Hugging Face](https://huggingface.co/).
2. Click on **Spaces** in the top navigation bar, then click **Create new Space**.
3. Set your Space name (e.g., `af-thermostability-cnn`).
4. Select **Gradio** as the SDK.
5. Choose **Public** or **Private** visibility, then click **Create Space**.

### B. Upload files to the Space
You can clone the Space repo locally via Git and copy all project files, or upload them directly via the Hugging Face Web UI:
1. Copy/upload these key files and folders:
   - `app.py`
   - `best_model.pth`
   - `requirements.txt`
   - `README.md` (contains Hugging Face metadata at the top)
   - `src/` (folder containing `model.py`, `predict.py`, `data_loader.py`)
   - `data/` (folder containing `synthetic/` examples for Gradio)
2. Commit and push the files. Hugging Face will automatically detect `app.py` and `requirements.txt`, install dependencies, and build/run the web app!

