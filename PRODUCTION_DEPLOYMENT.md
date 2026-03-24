# 🚀 DermAura — Production Deployment Guide (DigitalOcean)

This guide walks you through hosting your Dockerized application on a DigitalOcean Droplet using your new domain, **`dermaura.tech`**.

---

## 🏗 Phase 1: Server Setup (DigitalOcean)

### 1. Create a Droplet
- **Image**: Marketplace -> **Docker on Ubuntu** (includes Docker & Compose).
- **Size**: **Basic Plan (Regular SSD)**. 
- **RECOMMENDED**: **$12/mo (2GB RAM + 2 vCPU)**.
- **IMPORTANT**: To make this app run on 2GB or 4GB, you **must** follow the **Swap File** steps in Phase 1b.
- **Authentication**: SSH Keys (Recommended) or Password.

### 2. Connect via SSH
```bash
ssh root@your_droplet_ip
```

---

## 💾 Phase 1b: Memory Optimization (IMPORTANT)

Since your AI model is large, a 2GB server needs "Virtual Memory" (Swap) to prevent crashing on startup. 

**Run these commands on your server to create a 4GB Swap file:**
```bash
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab
```
*Wait 10 seconds for it to finish. You now have 2GB RAM + 4GB Virtual Memory (6GB total).*

---

## 🌐 Phase 2: Domain & DNS Setup

Log in to your domain registrar (where you bought `dermaura.tech`) and point it to your Droplet:

| Type | Host | Value | TTL |
| :--- | :--- | :--- | :--- |
| **A** | `@` | `your_droplet_ip` | Default |
| **A** | `www` | `your_droplet_ip` | Default |
| **A** | `api` | `your_droplet_ip` | Default |

---

## 🔒 Phase 3: Nginx & SSL (HTTPS)

We will use Nginx on the server as a "Reverse Proxy" to handle SSL and direct traffic to your containers.

### 1. Install Certbot
```bash
apt-get update
apt-get install certbot python3-certbot-nginx -y
```

### 2. Configure Nginx
Create a configuration file: `nano /etc/nginx/sites-available/dermaura`
```nginx
server {
    server_name dermaura.tech www.dermaura.tech;

    location / {
        proxy_pass http://localhost:3000; # Frontend container
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

server {
    server_name api.dermaura.tech;

    location / {
        proxy_pass http://localhost:8000; # Backend container
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
Enable it:
```bash
ln -s /etc/nginx/sites-available/dermaura /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx
```

### 3. Get SSL Certificates
```bash
certbot --nginx -d dermaura.tech -d www.dermaura.tech -d api.dermaura.tech
```
*Select option 2 (Redirect) when asked.*

---

## 🛠 Phase 4: Deploying with Docker

### 1. Clone & Configure
```bash
git clone https://github.com/your-username/SkinAi.git
cd SkinAi
nano .env
```

### 2. Production `.env` Settings
Update your `.env` to use the new domain:
```env
# ... existing MongoDB and Cloudinary keys ...

FRONTEND_URL="https://dermaura.tech"
VITE_API_URL="https://api.dermaura.tech"
```

### 3. Start the Application
```bash
docker-compose up -d --build
```

---

## ✅ You're Live!
- **Frontend**: [https://dermaura.tech](https://dermaura.tech)
- **Backend API**: [https://api.dermaura.tech](https://api.dermaura.tech)

---

## 🧹 Maintenance
- **Update code**: `git pull && docker-compose up -d --build`
- **View logs**: `docker-compose logs -f backend`
- **Renew SSL**: Certbot handles this automatically via cron.
