#!/usr/bin/env bash
# Reel Factory - Local LLM Setup Script
# Downloads and configures Ollama with all required models.
#
# Usage: ./setup_local_llms.sh
#
# Requirements: curl, tar, ~16GB disk space
set -e

INSTALL_DIR="$HOME/bin"
LIB_DIR="$HOME/lib/ollama"
MODELS_DIR="$HOME/.ollama/models"

echo "=== Reel Factory - Local LLM Setup ==="
echo ""

# 1. Check if Ollama is already installed
if command -v ollama &>/dev/null; then
    echo "[OK] Ollama already installed: $(ollama --version 2>&1 | grep -oP '\d+\.\d+\.\d+')"
elif [ -x "$INSTALL_DIR/ollama" ]; then
    echo "[OK] Ollama found at $INSTALL_DIR/ollama"
    export PATH="$INSTALL_DIR:$PATH"
else
    echo "[*] Installing Ollama..."

    # Detect architecture
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64) ARCH_SUFFIX="amd64" ;;
        aarch64|arm64) ARCH_SUFFIX="arm64" ;;
        *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
    esac

    # Download
    DOWNLOAD_URL="https://github.com/ollama/ollama/releases/latest/download/ollama-linux-${ARCH_SUFFIX}.tar.zst"
    echo "    Downloading from $DOWNLOAD_URL..."
    curl -fSL -o /tmp/ollama.tar.zst "$DOWNLOAD_URL"

    # Check for zstd
    if ! command -v zstd &>/dev/null; then
        echo "    zstd not found, attempting to install..."
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y -qq zstd 2>/dev/null || true
        fi
        # If still not available, try getting from deb
        if ! command -v zstd &>/dev/null; then
            echo "    Downloading zstd binary..."
            curl -fsSL -o /tmp/zstd.deb "http://archive.ubuntu.com/ubuntu/pool/main/libz/libzstd/zstd_1.5.5+dfsg2-2build1.1_amd64.deb" 2>/dev/null || true
            if [ -f /tmp/zstd.deb ]; then
                mkdir -p /tmp/zstd_ext && dpkg-deb -x /tmp/zstd.deb /tmp/zstd_ext
                mkdir -p "$INSTALL_DIR"
                cp /tmp/zstd_ext/usr/bin/zstd "$INSTALL_DIR/zstd"
                chmod +x "$INSTALL_DIR/zstd"
                export PATH="$INSTALL_DIR:$PATH"
            fi
        fi
    fi

    # Extract
    echo "    Extracting..."
    mkdir -p /tmp/ollama_extract
    tar --zstd -xf /tmp/ollama.tar.zst -C /tmp/ollama_extract

    # Install
    mkdir -p "$INSTALL_DIR" "$LIB_DIR"
    cp /tmp/ollama_extract/bin/ollama "$INSTALL_DIR/ollama"
    chmod +x "$INSTALL_DIR/ollama"
    cp -r /tmp/ollama_extract/lib/ollama/* "$LIB_DIR/" 2>/dev/null || true

    export PATH="$INSTALL_DIR:$PATH"
    echo "    Installed Ollama $(ollama --version 2>&1 | grep -oP '\d+\.\d+\.\d+' || echo 'OK')"

    # Cleanup
    rm -rf /tmp/ollama.tar.zst /tmp/ollama_extract /tmp/zstd.deb /tmp/zstd_ext
fi

# 2. Start Ollama server if not running
if curl -s http://localhost:11434/ >/dev/null 2>&1; then
    echo "[OK] Ollama server already running"
else
    echo "[*] Starting Ollama server..."
    export OLLAMA_LLM_LIBRARY="$LIB_DIR"
    export OLLAMA_MODELS="$MODELS_DIR"
    setsid ollama serve > /tmp/ollama_serve.log 2>&1 &
    disown
    sleep 3
    if curl -s http://localhost:11434/ >/dev/null 2>&1; then
        echo "    Server started successfully"
    else
        echo "    [!] Server failed to start. Check /tmp/ollama_serve.log"
        exit 1
    fi
fi

# 3. Pull required models
echo ""
echo "[*] Pulling required models (4 unique, ~15.5GB total)..."
echo ""

MODELS=(
    "llama3:8b"
    "mistral:7b-instruct"
    "qwen2:7b"
    "phi3:mini"
)

for model in "${MODELS[@]}"; do
    if ollama list 2>/dev/null | grep -q "^${model}"; then
        echo "    [OK] $model (already installed)"
    else
        echo "    [*] Pulling $model..."
        ollama pull "$model"
        echo "    [OK] $model pulled"
    fi
done

# 4. Verify
echo ""
echo "=== Setup Complete ==="
echo ""
ollama list
echo ""
echo "Models are ready for reel-factory!"
echo ""
echo "Usage:"
echo "  reel-factory create -s "Your script idea" -g comedy"
echo "  reel-factory models status"
echo ""
echo "To keep Ollama running permanently, add to your shell profile:"
echo "  export PATH="$INSTALL_DIR:\$PATH""
echo "  export OLLAMA_LLM_LIBRARY="$LIB_DIR""
echo "  export OLLAMA_MODELS="$MODELS_DIR""
