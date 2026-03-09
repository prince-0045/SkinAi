# Hosting & Deployment Guide for SkinAi

This guide explains how to host your SkinAi website and addresses your questions about model deployment and code changes.

## 1. Do we need to deploy the model?
**Yes.** The AI model ([model.h5](file:///d:/SkinAi/model.h5)) is the heart of your application. When you host the backend, the model file must be uploaded along with your code so the server can load it into memory to make predictions.

## 2. Potential Hosting Difficulties
*   **Model Size**: Your [model.h5](file:///d:/SkinAi/model.h5) is ~28MB. Free hosting tiers (like Render or Vercel) often have memory limits. TensorFlow/Keras can use a lot of RAM.
*   **Dependencies**: Installing `tensorflow` on a server can be slow and sometimes fails if the server has low memory. Using `tensorflow-cpu` is recommended for hosting to save space and memory.
*   **Environment Variables**: You must set your MongoDB URL, Secret Keys, and Cloudinary credentials in the hosting provider's dashboard.

## 4. Step-by-Step Deployment Guide

### PHASE 1: Deploy Backend on Render.com
1.  **Push to GitHub**: Make sure your `backend` folder and [model.h5](file:///d:/SkinAi/model.h5) are pushed to a GitHub repository.
2.  **Create Web Service**: In Render dashboard, click **New +** > **Web Service**.
3.  **Connect Repo**: Connect your GitHub repository.
4.  **Configure**:
    *   **Root Directory**: `backend`
    *   **Runtime**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
        *   *(Note: You do **not** need to type a number for `$PORT`. Render automatically replaces this with the correct port number when it starts your app.)*
5.  **Environment Variables**: Click "Advanced" or go to the "Environment" tab and add everything from your local [.env](file:///d:/SkinAi/.env):
    *   `MONGO_URL`: (Your MongoDB connection string)
    *   `SECRET_KEY`, `ALGORITHM`, etc.
    *   **IMPORTANT**: Set `FRONTEND_URL` later (once you have your Vercel URL).
6.  **Deploy**: Once it finishes, copy the URL Render gives you (e.g., `https://skinai-api.onrender.com`).

---

### PHASE 1 (Alternative): Deploy Backend on Railway.app
1.  **Push to GitHub**: (Done) Ensure the `backend` folder is at the root or correctly tracked.
2.  **Create New Project**: In Railway dashboard, click **+ New Project** > **Deploy from GitHub repo**.
3.  **Configure Service**:
    *   Railway usually auto-detects Python.
    *   Go to **Settings** > **Start Command** and enter:
        `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4.  **Variables**: Go to the **Variables** tab and add your [.env](file:///d:/SkinAi/.env) variables (e.g., `MONGO_URL`).
5.  **Generate Domain**: Go to **Settings** > **Public Networking** and click **Generate Domain** to get your API URL.

---

> [!TIP]
> **Fixing "Railpack could not determine how to build" on Railway**:
> 1. Go to your Railway service **Settings**.
> 2. Look for **Root Directory**. Set it to `/backend`.
> 3. If it still fails, I can create a `Procfile` for you inside the `backend` folder.

---

### PHASE 2: Deploy Frontend on Vercel.com
1.  **Import Repo**: In Vercel dashboard, click **Add New** > **Project**.
2.  **Select Repo**: Select the same GitHub repository.
3.  **Configure**:
    *   **Root Directory**: `frontend`
    *   **Framework Preset**: `Vite`
4.  **Environment Variables**:
    *   Add `VITE_API_URL` and paste your **Render URL** (e.g., `https://skinai-api.onrender.com`).
5.  **Deploy**: Click "Deploy".
6.  **Copy URL**: Once finished, Vercel will give you a URL (e.g., `https://skinai-frontend.vercel.app`).

---

### PHASE 3: Connect them Together
1.  **Update Backend CORS**: Go back to Render > your Service > Environment.
2.  **Set FRONTEND_URL**: Update the `FRONTEND_URL` variable to your **Vercel URL** (e.g., `https://skinai-frontend.vercel.app`).
3.  **Save/Re-deploy**: Render will automatically restart with the new setting.

## 5. Using Your Own Domain for Free?
**Yes!** Most modern hosting providers allow you to connect a custom domain (e.g., `www.yourname.com`) for free, even on their $0/month plans.

*   **Vercel/Netlify (Frontend)**: You can add your domain in the "Domain Settings" tab. They will provide you with DNS records (Nameservers or A/CNAME records) to point from your domain registrar to their servers.
*   **Render/Railway (Backend)**: They also allow custom domains for free. You can map `api.yourname.com` to your backend service.

> [!IMPORTANT]
> While the **hosting is free**, you still need to **buy the domain name** itself from a registrar (like Namecheap, GoDaddy, or Cloudflare). Domain names usually cost ~$10-$20 per year.

## 6. Alternative Free Backend Hosting
If Render is slow or has issues, here are other free options for your Python backend:

1.  **Railway.app**: Very easy to use. Offers $5 of free credit (enough for a month or two of testing). It is often faster than Render for AI models.
2.  **Koyeb**: Has a "Free Tier" (Nano instance). It is great for low-traffic sites and very simple to connect to GitHub.
3.  **Hugging Face Spaces**: Since you are using a Machine Learning model, you can host your API as a **Docker Space** on Hugging Face for free. This is excellent for Tensorbridge/ML apps.
4.  **Google Cloud Run**: Has a generous "Free Tier" but requires a credit card to sign up.

> [!TIP]
> **TensorFlow Note**: If your server runs out of memory (RAM), try using `tensorflow-cpu` in your `requirements.txt` instead of the full `tensorflow`. It is much smaller and uses less memory.

## 7. Impact of Changes
*   **Changing Model**: If you replace `model.h5`, you must re-upload/re-deploy the backend. The website will then use the new model immediately.
*   **Editing Code/Features**: Any change to the code requires a new "push" to GitHub, which usually triggers an automatic re-deployment.
*   **Effect on Website**: Changes are safe but will cause a brief downtime (seconds) while the new version starts up.

## 5. Summary of Bugs Found
1.  **Hardcoded Paths**: The backend looks specifically for `D:\SkinAi\model.h5`. This will crash on a Linux server.
2.  **Fixed CORS**: The backend origin is limited to localhost. It needs to allow your production frontend URL.
3.  **Frontend API Proxy**: The frontend relies on a development proxy that won't work in production without configuration.
