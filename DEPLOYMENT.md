# Deployment Guide for Finger Alarm Nightmare

This guide explains how to deploy your Streamlit app to **Render** and keep it awake.

## 1. Deploying on Render (Private Repo)

Render is reliable for private repos.

1. **Sign Up**: Go to [render.com](https://render.com/).
2. **Create Service**: Click **"New +"** -> **"Web Service"**.
3. **Connect Repo**: Select your **private repository**.
4. **Settings**:
   - **Runtime**: Select **Docker**.
   - **Instance Type**: Select **Free**.
5. **Environment Variables**:
   - Key: `DEV_MODE` | Value: `false`
   - Key: `PORT` | Value: `8501`
6. **Deploy**: Click **"Create Web Service"**.

### ⚠️ Critical Port Setting

Render maps port 10000 by default. You **MUST** change this effectively.
In the Render Dashboard for your service:

1. Go to **Settings** -> **Networking**.
2. Set **Port** to `8501`.
3. Save.

---

## 2. Preventing "Sleep" (UptimeRobot)

Render's free tier spins down after inactivity. To keep it alive, use a ping service.

**We will use the built-in Streamlit Health Endpoint:**
Your app already has a lightweight "health check" URL at:  
`https://<YOUR-APP-NAME>.onrender.com/_stcore/health`

This endpoint returns `OK` immediately without loading the heavy video libraries, making it perfect for pings.

### Setup Instructions:

1. Go to [uptimerobot.com](https://uptimerobot.com/) and create a free account.
2. Click **"Add New Monitor"**.
3. **Monitor Type**: Select **HTTP(s)**.
4. **Friendly Name**: e.g., "Finger Alarm".
5. **URL (or IP)**: `https://<YOUR-APP-NAME>.onrender.com/_stcore/health`
   _(Replace `<YOUR-APP-NAME>` with your actual Render URL)_.
6. **Monitoring Interval**: Set to **5 minutes** (this is frequent enough to prevent sleep).
7. **Create Monitor**.

That's it! UptimeRobot will ping your app every 5 minutes, ensuring Render considers it "active" and doesn't spin it down.

---

## Troubleshooting

### "Check Health" Failures on Render Deployment

If the deploy fails during health checks:

- Ensure the **Port** is set to `8501`.
- Check logs for "ModuleNotFoundError".

### Camera/WebRTC Issues

If video doesn't load:

- It might be a firewall issue. The deployed app uses Google's public STUN server.
- If it works on mobile data but not Wi-Fi, it's a network restriction (common in schools/offices).
