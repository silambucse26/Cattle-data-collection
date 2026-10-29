#!/bin/bash
set -e

echo "=========================================================="
echo " Starting PashuX AI Production Deployment Setup (24/7 Cloud)"
echo "=========================================================="

# 1. Update system packages
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv git ffmpeg nginx libgl1-mesa-glx libglib2.0-0 curl

# 2. Setup Virtual Environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create required directories
mkdir -p captures/images captures/videos uploads

# 4. Configure Systemd Service for 24/7 background running
CURRENT_USER=$(whoami)
CURRENT_DIR=$(pwd)

sudo sed -i "s|User=ubuntu|User=$CURRENT_USER|g" deploy/systemd/pashux.service
sudo sed -i "s|/home/ubuntu/Cattle-data-collection|$CURRENT_DIR|g" deploy/systemd/pashux.service

sudo cp deploy/systemd/pashux.service /etc/systemd/system/pashux.service
sudo systemctl daemon-reload
sudo systemctl enable pashux
sudo systemctl restart pashux

# 5. Configure Nginx Reverse Proxy
sudo cp deploy/nginx/pashux.conf /etc/nginx/sites-available/pashux.conf
sudo ln -sf /etc/nginx/sites-available/pashux.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

echo "=========================================================="
echo " PashuX AI is now running 24/7 in production!"
echo " Status: sudo systemctl status pashux"
echo " Logs:   sudo journalctl -u pashux -f"
echo "=========================================================="
