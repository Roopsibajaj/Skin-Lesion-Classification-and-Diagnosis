# Running the Skin Lesion Classifier Locally

Follow these instructions to set up and run the project on your local machine.

---

## 📋 Prerequisites

- **Python 3.9+** installed on your system.
- An internet connection (if you need to download datasets or missing dependencies).
- The project already includes a virtual environment folder named `venv` with TensorFlow and Flask pre-installed.

---

## 🚀 Quick Start Steps

### Step 1: Open Terminal / Command Prompt
Open your terminal and navigate to the project directory:
```bash
cd "/Users/aniketbhatia/untitled folder 2/skin-lesion-classifier"
```

### Step 2: Activate the Virtual Environment
Activate the pre-existing virtual environment (`venv`) depending on your operating system:

* **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```
* **Windows (Command Prompt)**:
  ```cmd
  venv\Scripts\activate
  ```
* **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

Once activated, your terminal prompt will show `(venv)` at the beginning.

### Step 3: Run the Web Server
Start the Flask application:
```bash
python app.py
```

You should see output indicating that the server is running, for example:
```
  SKIN LESION CLASSIFICATION — WEB APPLICATION
============================================================
  Server:  http://127.0.0.1:5000
  Model:   models/skin_lesion_model.h5
  Uploads: uploads
============================================================
```

### Step 4: Open in Web Browser
Open your browser and visit:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 🛠 Troubleshooting & Other Scripts

### Missing Dependencies
If you encounter import errors or are setting up on a new machine, reinstall the required packages with the virtual environment activated:
```bash
pip install -r requirements.txt
```

### Re-creating a Dummy Model (For Testing)
If the model file `models/skin_lesion_model.h5` is ever deleted or missing, you can create a fast dummy model for testing the interface without waiting for the full dataset training:
```bash
python create_dummy_model.py
```

### Training the Full Model
If you have the dataset downloaded and want to run the full training pipeline:
```bash
python -m src.train
```
