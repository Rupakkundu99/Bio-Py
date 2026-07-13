# End-to-End Hugging Face Deployment Guide

This guide explains how to host your protein thermostability prediction app on Hugging Face Spaces so that anyone in the world can open it in their browser and use it.

---

## 1. How Our App Works (Frontend and Backend)

Before deploying, it helps to understand what we are hosting:

*   **The Frontend**: This is the visual webpage (the upload buttons, tabs, sliders, and charts) that you see and click on. It is created using the Gradio library in Python.
*   **The Backend**: This is the engine under the hood. It includes the PyTorch library loading your trained model weights (`best_model.pth`), parsing the PDB structure files into 2D matrices, calculating biophysical heuristics, and sending requests to the Google Gemini API for explanations.

In our project, **`app.py` contains both the frontend and the backend**. When a user uploads a file, the frontend captures it and immediately passes it to the backend functions in the same file to run the calculations.

### Do I have to host them separately?
**No. They are hosted together in the exact same Space.** Hugging Face will run your `app.py` script as a single process. It handles starting the backend engine and automatically exposing the frontend webpage to the public. Anyone you share the Space link with can visit the page and use the app directly in their browser.

---

## 2. Step 1: Create a Hugging Face Account

1. Open your web browser and go to [Hugging Face](https://huggingface.co/).
2. Click the **Sign Up** button in the top right corner.
3. Fill in your email, choose a password, and verify your account via the email confirmation link they send you.

---

## 3. Step 2: Create a New Space

A "Space" is a free hosting container provided by Hugging Face to run interactive web applications.

1. Once logged in, click on your profile picture in the top-right corner of the Hugging Face homepage.
2. Select **New Space** from the menu.
3. Fill out the configuration page:
    *   **Space Name**: Type a name (for example, `protein-thermostability-cnn`).
    *   **SDK (Software Development Kit)**: Select **Gradio**.
    *   **Gradio Template**: Choose **Blank** (do not select a pre-configured template).
    *   **Space Hardware**: Select the default free tier (**CPU Basic - 2 vCPU, 16GB RAM**). The model is lightweight and runs fast on a CPU.
    *   **Visibility**: Choose **Public** (so anyone can visit) or **Private** (only you can see it).
4. Click the **Create Space** button at the very bottom.

---

## 4. Step 3: Securely Add Your Gemini API Key

We want the app to have access to your Google Gemini API Key so it can generate AI explanations, but we must **never** write the key directly in the code (otherwise, other people could steal it). Hugging Face provides a safe place to store keys called **Secrets**.

1. On your newly created Space webpage, look at the top menu tabs and click on **Settings** (usually next to Files).
2. Scroll down until you see the section called **Variables and secrets**.
3. Click the **New secret** button.
4. Fill in the fields:
    *   **Name**: Type exactly `GEMINI_API_KEY`.
    *   **Value**: Paste your actual Google Gemini API Key (e.g., `AIzaSy...`).
5. Click **Save**. 

Our app is programmed to look for this environment secret automatically. Visitors will be able to get AI explanations instantly without needing to paste a key themselves, and your key will remain completely hidden from everyone.

---

## 5. Step 4: Upload Your Files

You need to copy the project files from your computer to your Hugging Face Space repository. You can do this directly inside your browser:

1. On your Space page, click the **Files** tab (next to App).
2. Click the **Add file** button in the top right, then select **Upload files**.
3. Drag and drop the following files and folders from your project directory:
    *   `app.py` (the main web app file)
    *   `best_model.pth` (the trained neural network model weights)
    *   `requirements.txt` (the list of libraries Hugging Face needs to install)
    *   `README.md` (contains the Hugging Face settings header)
    *   `.gitignore` (tells Git to ignore temporary files)
    *   `src/` (the folder containing `model.py`, `predict.py`, and `data_loader.py`)
    *   `data/` (the folder including the `synthetic/` subfolder containing the example PDB/PAE files)

> [!NOTE]
> **What about `__pycache__` and other junk files?**
> *   If you are using **Git** to upload, the project already has a `.gitignore` file that automatically blocks `__pycache__/` and other compiled python files from being uploaded.
> *   If you are using the **Web Browser drag-and-drop**, simply avoid dragging the `__pycache__/` folder (located inside `src/`) and the `venv/` or `.venv/` folders. If you accidentally drag them, you can delete them from the file list before committing.
> *   Never upload the `.env` file either, as your API key is now securely stored in the Space Secrets.

4. Scroll down to the bottom of the page, type a commit message (like `Initial upload`), and click **Commit changes to main**.

---

## 6. Step 5: Build and Run

As soon as you commit the files, Hugging Face will automatically start setting up your application:

1. Click on the **App** tab at the top of your Space.
2. You will see a status badge saying **Building** or **Running**. Hugging Face is currently reading your `requirements.txt` file, downloading PyTorch, Gradio, and other required packages, and setting up the server.
3. This setup phase can take 2 to 5 minutes the first time.
4. Once the status badge changes to a green **Running**, your interface will appear.

You can now upload PDB files, click on example structures, or enter a UniProt ID to run predictions. If you added the `GEMINI_API_KEY` secret in Step 3, clicking **Generate AI Biophysics Explanation** will instantly generate the AI analysis report.
