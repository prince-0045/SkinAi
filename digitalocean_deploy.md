# Deploying Your Dockerized App to DigitalOcean

Yes, you can absolutely deploy this directly to DigitalOcean since your Docker containers are fully ready! It is one of the most cost-effective and straightforward ways to host a Docker Compose application.

Here is your step-by-step guide to doing exactly that:

## 1. Prepare Your Code
Before putting it on the server, the easiest way to transfer your code is via GitHub.
1. Make sure your latest code (including our recent limit and UI fixes) is pushed to your GitHub repository.

## 2. Create a DigitalOcean Droplet
A "Droplet" is just a virtual server. DigitalOcean has a pre-built image that comes with Docker and Docker Compose already installed!
1. Log in to your DigitalOcean account.
2. Click the green **Create** button at the top right, and select **Droplets**.
3. **Choose Region**: Pick the datacenter closest to you or your users.
4. **Choose an Image**: Click the **Marketplace** tab, search for **Docker**, and select the official Docker app (this saves you from installing Docker manually).
5. **Choose Core Plan**: Select **Basic**. Because your app runs Machine Learning models (`model.h5`), you will likely run out of memory on the $4/mo plan. I highly recommend at least the **$6/mo or $12/mo plan** (2GB RAM).
6. **Authentication Method**: Choose **Password** (and create a strong password) OR **SSH Key** (more secure).
7. Click **Create Droplet** at the bottom.

## 3. Connect to Your Server
1. Once the Droplet is created, copy its **IPv4 address**.
2. Open your terminal/command prompt on your local computer and run:
   ```bash
   ssh root@<YOUR_DROPLET_IP_ADDRESS>
   ```
3. Type `yes` if it asks about the fingerprint, and enter the password you created.

## 4. Download Your Code to the Server
Now that you are inside your server's terminal:
1. Clone your GitHub repository:
   ```bash
   git clone <YOUR_GITHUB_REPO_URL>
   ```
2. Move into your project folder:
   ```bash
   cd <YOUR_REPO_NAME>
   ```

## 5. Set Up Your Environment Variables
Because `.env` files contain secrets, they usually aren't securely uploaded to GitHub. You need to recreate it on the server.
1. Open a new file editor:
   ```bash
   nano .env
   ```
2. Paste all your `.env` contents (Cloudinary URLs, MongoDB connection strings, SECRET_KEY, etc.) into this file.
3. **Crucial Step**: In your `docker-compose.yml`, your frontend looks for `VITE_API_URL`. Ensure you update this in your `.env` (or directly in docker-compose.yml if hardcoded) to point to your new server's IP instead of localhost:
   `VITE_API_URL=http://<YOUR_DROPLET_IP_ADDRESS>:8000`
4. Press `CTRL+X`, then `Y`, then `Enter` to save the file.

## 6. Run Your App
1. With your code and `.env` ready, run the exact same command you did locally:
   ```bash
   docker-compose up -d --build
   ```
2. Docker will download all the necessary Python and Node.js dependencies, build your frontend, and start everything up!

## 7. You're Live!
Once the command finishes, open your browser and go to your server's IP address on port 3000:
**`http://<YOUR_DROPLET_IP_ADDRESS>:3000`**

Your Skin AI detection app is now live on the internet! 

*Note: Since you mapped port 3000 in your docker-compose, you must type `:3000` at the end of the IP. If you ever want to link a real domain name (like `www.skinai.com`), you can use a reverse proxy like Nginx or Cloudflare in front of it later.*
