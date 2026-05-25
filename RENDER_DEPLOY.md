# Deploying to Render

This guide outlines how to deploy the Skin Lesion Classifier web application to **Render** (https://render.com).

---

## 🛠 Prerequisites

1. A **Render** account (Free tier works, but 512MB RAM is close to the limit for TensorFlow).
2. The code pushed to your GitHub repository (e.g., `https://github.com/Roopsibajaj/Skin-Lesion-Classification-and-Diagnosis.git`).

---

## 🚀 Step-by-Step Deployment Guide

### Step 1: Create a New Web Service on Render
1. Log in to the [Render Dashboard](https://dashboard.render.com).
2. Click **New** (top-right) and select **Web Service**.
3. Connect your GitHub account (if not already done) and select the **Skin-Lesion-Classification-and-Diagnosis** repository.

### Step 2: Configure Web Service Settings
Fill in the deployment details:

* **Name**: `skin-lesion-classifier` (or any name you prefer)
* **Region**: Choose the region closest to you (e.g., `Oregon (US West)` or `Frankfurt (EU Central)`)
* **Branch**: `main`
* **Root Directory**: *Leave empty* (if the project is in the root of the repository)
* **Runtime**: `Python 3`
* **Build Command**: 
  ```bash
  pip install -r requirements.txt
  ```
* **Start Command**: 
  ```bash
  gunicorn app:app
  ```

### Step 3: Choose Instance Type
- Choose the **Free** or **Individual/Starter** tier.
- *Note:* TensorFlow requires significant RAM to load. On the **Free** tier (512MB RAM), the application might occasionally exceed the memory limit and restart when loading the model for the first time. We have optimized `requirements.txt` to install `tensorflow-cpu` on Render, which uses much less memory than standard TensorFlow.

### Step 4: Click Deploy!
Render will now pull the code from GitHub, install the dependencies, and start the Gunicorn web server.

Once the logs show `[INFO] Booting worker with pid: ...`, the service is live! You can access the URL provided by Render (e.g., `https://skin-lesion-classifier.onrender.com`).

---

## 💡 Important Production Considerations on Render

1. **Ephemeral File System**: 
   Render Web Services have an ephemeral filesystem. This means any images uploaded by users into the `uploads/` folder will be deleted whenever the server restarts or deploys a new version. This is perfectly fine for a demo, but for a true production system, you should connect to a cloud storage provider (like AWS S3) to store uploads.
   
2. **First Request Latency (Cold Starts)**: 
   If using the Render Free tier, the web service will "spin down" after 15 minutes of inactivity. When a new user visits, it will trigger a "cold start" which can take 1–2 minutes to spin up the server and load the deep learning model.
