#!/usr/bin/env bash
# =============================================================================
# uniserve-fasth3-easy-install.sh
#
# One-shot installer for UniServe + FastH3 NVFP4 on a single NVIDIA GPU.
# Validated on: RTX 5090 (32 GB), RTX PRO 6000 (96 GB), B200 (192 GB).
# Uses the DEFAULT Hugging Face cache (~/.cache/huggingface/hub).
#
# See README.md for full documentation.
#
# Usage:
#   chmod +x uniserve-fasth3-easy-install.sh
#   ./uniserve-fasth3-easy-install.sh             # install only
#   ./uniserve-fasth3-easy-install.sh --serve     # install + start server
#   ./uniserve-fasth3-easy-install.sh --skip-cuda # CUDA 13 already installed
# =============================================================================

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
UNISERVE_DIR="$WORKDIR/UniServe"
MODEL_REPO="FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4"
SERVED_NAME="FastH3"
PORT="${PORT:-8000}"
LOG_FILE="${LOG_FILE:-$WORKDIR/install-uniserve.log}"

mkdir -p "$WORKDIR"
exec > >(tee -a "$LOG_FILE") 2>&1

SKIP_CUDA=false
DO_SERVE=false
for arg in "$@"; do
    case "$arg" in
        --skip-cuda) SKIP_CUDA=true ;;
        --serve)     DO_SERVE=true ;;
        --help|-h)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ---------- Pretty output ----------
BOLD=$'\033[1m'; CYAN=$'\033[1;36m'; GREEN=$'\033[1;32m'
YELLOW=$'\033[1;33m'; RED=$'\033[1;31m'; RESET=$'\033[0m'

ts() { date +%H:%M:%S; }
log()  { printf '\n%s[%s] %s%s\n' "$CYAN" "$(ts)" "$*" "$RESET"; }
ok()   { printf '%s[%s] ✓ %s%s\n' "$GREEN" "$(ts)" "$*" "$RESET"; }
warn() { printf '\n%s[%s] WARN: %s%s\n' "$YELLOW" "$(ts)" "$*" "$RESET"; }
die()  { printf '\n%s[%s] FAIL: %s%s\n' "$RED" "$(ts)" "$*" "$RESET" >&2; exit 1; }

step() {
    local n="$1"; shift
    printf '\n%s────────────────────────────────────────────────────────────%s\n' "$BOLD" "$RESET"
    printf '%s STEP %s — %s  [%s]%s\n' "$BOLD$CYAN" "$n" "$*" "$(ts)" "$RESET"
    printf '%s────────────────────────────────────────────────────────────%s\n' "$BOLD" "$RESET"
}

run() {
    local label="$1"; shift
    printf '\n%s▶ %s%s\n' "$BOLD" "$label" "$RESET"
    printf '%s  $ %s%s\n' "$YELLOW" "$*" "$RESET"
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        die "command failed (rc=$rc): $*"
    fi
}

# =============================================================================
step "0" "Preflight"
# =============================================================================
log "Log file: $LOG_FILE"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi missing"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
ok "GPU: $GPU_NAME ($GPU_MEM)"

FREE_GB=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
ok "Free on /: ${FREE_GB} GB"
[ "$FREE_GB" -ge 130 ] || warn "Less than 130 GB — model + venv need ~110 GB"

# =============================================================================
step "1" "CUDA 13.0 toolkit"
# =============================================================================
NEED_CUDA=true
if [ "$SKIP_CUDA" = true ]; then
    ok "Skipped (--skip-cuda)"
    NEED_CUDA=false
elif command -v nvcc >/dev/null 2>&1; then
    CUR=$(nvcc --version | grep -oP 'release \K[0-9.]+' || echo "0")
    log "Existing nvcc: $CUR"
    if [[ "$CUR" == 13.* ]]; then
        ok "CUDA 13 already present"
        NEED_CUDA=false
    fi
fi

if [ "$NEED_CUDA" = true ]; then
    TMP=$(mktemp -d)
    cd "$TMP"

    log "Fetching CUDA repo pin"
    run "download pin" \
        wget -q --show-progress \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin \
        -O cuda-ubuntu2404.pin
    sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600

    log "Downloading CUDA 13.0 local repo (~200 MB)"
    run "download cuda-repo.deb" \
        wget -q --show-progress \
        https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb \
        -O cuda-repo.deb

    log "Installing local repo package"
    run "dpkg -i cuda-repo.deb" sudo dpkg -i cuda-repo.deb

    run "install keyring" \
        sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/

    run "apt-get update" sudo apt-get update -qq

    log "Installing cuda-toolkit-13-0 (~3 GB, watch progress below)"
    run "apt-get install cuda-toolkit-13-0" \
        sudo apt-get -y install cuda-toolkit-13-0

    cd "$WORKDIR"
    rm -rf "$TMP"
    ok "CUDA 13 installed"
fi

export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-13.0

if ! grep -q 'cuda-13.0' ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<'RC'

# --- CUDA 13 ---
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-13.0
RC
fi

if command -v nvcc >/dev/null 2>&1; then
    ok "nvcc: $(nvcc --version | grep -oP 'release \K[0-9.]+')"
fi

# =============================================================================
step "2" "Install uv + Rust"
# =============================================================================
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv..."
    run "install uv" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
else
    ok "uv already installed"
fi

if ! command -v rustc >/dev/null 2>&1; then
    log "Installing Rust (rustup will show progress below)"
    run "install rust" bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable'
else
    ok "rustc already installed"
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! grep -q '.cargo/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

command -v uv    >/dev/null 2>&1 || die "uv missing after install"
command -v rustc >/dev/null 2>&1 || die "rustc missing after install"

ok "uv:    $(uv --version)"
ok "rustc: $(rustc --version)"

# =============================================================================
step "3" "Clone + build UniServe"
# =============================================================================
if [ -d "$UNISERVE_DIR/.git" ]; then
    log "Updating existing UniServe clone"
    cd "$UNISERVE_DIR"
    run "git fetch" git fetch --depth=1 origin
    run "git reset" bash -c 'git reset --hard origin/HEAD 2>/dev/null || git reset --hard origin/main'
else
    cd "$WORKDIR"
    log "Cloning hao-ai-lab/UniServe (progress below)"
    run "git clone" git clone --progress https://github.com/hao-ai-lab/UniServe.git
    cd "$UNISERVE_DIR"
fi

log "Building UniServe — this is the longest step (10-20 min)"
log "Cargo output will stream below:"
run "uv sync (build)" \
    env UV_TORCH_BACKEND=cu130 uv sync --locked --python /usr/bin/python3.12 --extra gpu

source .venv/bin/activate

command -v uniserve >/dev/null 2>&1 || die "uniserve not in venv after build"
ok "uniserve: $(which uniserve)"

# =============================================================================
step "4" "Download FastH3 model (default HF cache)"
# =============================================================================
HF_DEFAULT="$HOME/.cache/huggingface"
mkdir -p "$HF_DEFAULT"

FREE_HOME_GB=$(df -BG "$HF_DEFAULT" | tail -1 | awk '{print $4}' | tr -d 'G')
log "HF cache: $HF_DEFAULT (${FREE_HOME_GB} GB free)"

if [ "$FREE_HOME_GB" -lt 100 ]; then
    warn "Only ${FREE_HOME_GB} GB at HF cache; model needs ~90 GB."
    warn "Redirect: export HF_HOME=/workspace/.hf_home  (before rerunning)"
fi

if ! hf auth whoami >/dev/null 2>&1; then
    warn "Not logged into HF — running 'hf auth login' now"
    hf auth login || true
fi

log "Downloading $MODEL_REPO"
run "hf download" hf download "$MODEL_REPO"

SNAPSHOT_DIR=$(find "$HF_DEFAULT/hub/models--FastVideo--FastVideo-FastH3-8-Step-V2-NVFP4/snapshots" \
    -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)

[ -n "$SNAPSHOT_DIR" ] || die "model snapshot not found after download"
ok "Model: $SNAPSHOT_DIR"
du -sh "$SNAPSHOT_DIR"

# =============================================================================
step "5" "Locate single-GPU config"
# =============================================================================
CONFIG=""
for c in \
    "$UNISERVE_DIR/configs/minimax_h3/fasth3-single.json" \
    "$UNISERVE_DIR/configs/minimax_h3/fasth3_1gpu.json" \
    "$UNISERVE_DIR/configs/minimax_h3/rtx5090.json" \
    "$UNISERVE_DIR/configs/minimax_h3/single.json" \
; do
    [ -f "$c" ] && CONFIG="$c" && break
done

if [ -z "$CONFIG" ]; then
    warn "No single-GPU config found. Available:"
    ls -1 "$UNISERVE_DIR/configs/minimax_h3/" 2>/dev/null | sed 's/^/    /' || true
else
    ok "Config: $CONFIG"
fi

# =============================================================================
step "6" "Done"
# =============================================================================
cat <<EOF

${GREEN}════════════════════════════════════════════════════════════════${RESET}
${GREEN}  UniServe + FastH3 ready${RESET}
${GREEN}════════════════════════════════════════════════════════════════${RESET}

  UniServe:  $UNISERVE_DIR
  Model:     $SNAPSHOT_DIR
  Config:    ${CONFIG:-<default>}
  Port:      $PORT
  Log:       $LOG_FILE

  To serve:

    cd $UNISERVE_DIR
    source .venv/bin/activate
    uniserve serve "$SNAPSHOT_DIR" \\
      ${CONFIG:+--workers "$CONFIG" \\}
      --served-model-name $SERVED_NAME \\
      --host 0.0.0.0 \\
      --port $PORT \\
      --max-running-requests 1

  First start: 10-15 min warmup.

${GREEN}════════════════════════════════════════════════════════════════${RESET}
EOF

# =============================================================================
step "7" "Serve (live)"
# =============================================================================
if [ "$DO_SERVE" = true ]; then
    log "Launching UniServe — streaming log below"
    log "Watch for 'worker ready' or 'listening on 0.0.0.0:$PORT'"
    cd "$UNISERVE_DIR"
    source .venv/bin/activate

    if [ -n "$CONFIG" ]; then
        exec uniserve serve "$SNAPSHOT_DIR" \
            --workers "$CONFIG" \
            --served-model-name "$SERVED_NAME" \
            --host 0.0.0.0 \
            --port "$PORT" \
            --max-running-requests 1
    else
        exec uniserve serve "$SNAPSHOT_DIR" \
            --served-model-name "$SERVED_NAME" \
            --host 0.0.0.0 \
            --port "$PORT" \
            --max-running-requests 1
    fi
fi
