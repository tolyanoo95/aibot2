#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  AI Crypto Futures Scanner — Server Deploy Script
#  Run on a fresh Ubuntu 22+ VPS as root:
#    bash deploy.sh
# ═══════════════════════════════════════════════════════════

set -e

APP_DIR="/root/aibot"
VENV="$APP_DIR/venv"
SERVICE="aibot"

echo "═══════════════════════════════════════════"
echo "  AI Crypto Bot — Deploy"
echo "═══════════════════════════════════════════"

# ── 1. System packages ───────────────────────────────────
echo "[1/7] Installing system packages..."
apt update -qq
apt install -y python3 python3-pip python3-venv git curl > /dev/null 2>&1
echo "  ✓ System packages installed"

# ── 2. Create app directory ──────────────────────────────
echo "[2/7] Setting up app directory..."
mkdir -p "$APP_DIR/models"

# Copy project files (this script should be run from the project dir)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ "$SCRIPT_DIR" != "$APP_DIR" ]; then
    cp -r "$SCRIPT_DIR"/src "$APP_DIR/"
    cp "$SCRIPT_DIR"/main.py "$APP_DIR/"
    cp "$SCRIPT_DIR"/train.py "$APP_DIR/"
    cp "$SCRIPT_DIR"/auto_retrain.py "$APP_DIR/"
    cp "$SCRIPT_DIR"/backtest.py "$APP_DIR/"
    cp "$SCRIPT_DIR"/backtest_llm.py "$APP_DIR/"
    cp "$SCRIPT_DIR"/requirements.txt "$APP_DIR/"
    cp "$SCRIPT_DIR"/.env.example "$APP_DIR/"
    cp "$SCRIPT_DIR"/.gitignore "$APP_DIR/" 2>/dev/null || true
    # copy model if exists
    cp "$SCRIPT_DIR"/models/*.json "$APP_DIR/models/" 2>/dev/null || true
    cp "$SCRIPT_DIR"/models/*.pkl "$APP_DIR/models/" 2>/dev/null || true
fi
echo "  ✓ Files copied to $APP_DIR"

# ── 3. Python venv ───────────────────────────────────────
echo "[3/7] Creating Python virtual environment..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install -r "$APP_DIR/requirements.txt" -q
echo "  ✓ Python venv ready ($(python3 --version))"

# ── 4. Environment file ─────────────────────────────────
echo "[4/7] Setting up .env..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo ""
    echo "  ╔══════════════════════════════════════════════╗"
    echo "  ║  IMPORTANT: Edit .env with your API keys!   ║"
    echo "  ║  nano $APP_DIR/.env                         ║"
    echo "  ╚══════════════════════════════════════════════╝"
    echo ""
else
    echo "  ✓ .env already exists (keeping current)"
fi

# ── 5. Train model (if not already trained) ──────────────
if [ ! -f "$APP_DIR/models/signal_model.json" ]; then
    echo "[5/7] Training ML model (first time, ~3-5 min)..."
    cd "$APP_DIR"
    "$VENV/bin/python" train.py
    echo "  ✓ Model trained"
else
    echo "[5/7] ML model already exists, skipping training"
    echo "  ✓ Model found at $APP_DIR/models/signal_model.json"
fi

# ── 6. Systemd service ──────────────────────────────────
echo "[6/7] Creating systemd service..."
cat > /etc/systemd/system/${SERVICE}.service << 'UNIT'
[Unit]
Description=AI Crypto Futures Scanner
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/aibot
ExecStart=/root/aibot/venv/bin/python main.py
Restart=always
RestartSec=30
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable ${SERVICE} > /dev/null 2>&1
echo "  ✓ Systemd service created"

# ── 7. Cron for auto-retrain ────────────────────────────
echo "[7/7] Setting up auto-retrain cron..."
CRON_LINE="0 3 */2 * * cd /root/aibot && /root/aibot/venv/bin/python auto_retrain.py >> /root/aibot/retrain.log 2>&1"
(crontab -l 2>/dev/null | grep -v "auto_retrain"; echo "$CRON_LINE") | crontab -
echo "  ✓ Auto-retrain: every 2 days at 03:00 UTC"

# ── Done ─────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  Deploy complete!"
echo "═══════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Edit API keys:"
echo "     nano $APP_DIR/.env"
echo ""
echo "  2. Start the bot:"
echo "     systemctl start $SERVICE"
echo ""
echo "  3. Check status:"
echo "     systemctl status $SERVICE"
echo ""
echo "  4. Watch logs:"
echo "     tail -f $APP_DIR/scanner.log"
echo "     # or"
echo "     journalctl -u $SERVICE -f"
echo ""
echo "  5. Stop the bot:"
echo "     systemctl stop $SERVICE"
echo ""
echo "  6. Restart after .env changes:"
echo "     systemctl restart $SERVICE"
echo ""
echo "  7. Run backtest:"
echo "     cd $APP_DIR && $VENV/bin/python backtest.py"
echo ""
echo "  8. Manual retrain:"
echo "     cd $APP_DIR && $VENV/bin/python train.py"
echo ""
