# UniServe + FastH3 Easy Install

One-shot installer for **UniServe** serving the **FastH3 8-step distilled model** on a single NVIDIA GPU. Tuned for batch video generation where per-clip time matters.

Tested on:
- **RTX 5090** (32 GB) — 14.8 s/clip at 480p, 38.6 s at 768p
- **RTX PRO 6000 Blackwell** (96 GB) — 13.5 s/clip at 480p, 36.5 s at 768p
- **B200** (192 GB) — under 10 s/clip at 480p

---

## What it installs

| Component | Purpose |
|---|---|
| **CUDA 13.0 toolkit** | Required by UniServe's custom CUDA kernels |
| **uv** | Python package manager used by UniServe |
| **Rust (stable)** | UniServe's server is written in Rust |
| **UniServe** | Serving engine for FastH3 with an OpenAI-compatible video API |
| **FastH3 NVFP4** | Distilled 8-step MiniMax-H3 variant (~90 GB) |

All files land in their standard locations:
- UniServe source: `/workspace/UniServe`
- Model: `~/.cache/huggingface/hub/models--FastVideo--FastVideo-FastH3-8-Step-V2-NVFP4/`
- Install log: `/workspace/install-uniserve.log`

---

## Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| **GPU VRAM** | 32 GB (RTX 5090) | 96 GB (RTX PRO 6000) |
| **System RAM** | 64 GB | 128 GB |
| **Disk** | 130 GB free | 250 GB |
| **CUDA driver** | 580+ | 580+ |
| **OS** | Ubuntu 24.04 | Ubuntu 24.04 |
| **Python** | 3.12 (`/usr/bin/python3.12`) | 3.12 |

The installer refuses to proceed if `nvidia-smi` is missing. It warns (but continues) if free disk is below 130 GB.

---

## Usage

```bash
# 1. Download and make executable
chmod +x uniserve-fasth3-easy-install.sh

# 2. Run the installer
./uniserve-fasth3-easy-install.sh
```

The script streams every subprocess output live to your terminal and mirrors it to `/workspace/install-uniserve.log`.

### Flags

| Flag | Effect |
|---|---|
| `--serve` | After install, immediately start the FastH3 server on port 8000 |
| `--skip-cuda` | Skip CUDA 13 install (use if already installed) |
| `--help`, `-h` | Show usage |

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `WORKDIR` | `/workspace` | Where UniServe is cloned |
| `UNISERVE_DIR` | `$WORKDIR/UniServe` | Override clone location |
| `PORT` | `8000` | Server port |
| `LOG_FILE` | `$WORKDIR/install-uniserve.log` | Install log path |
| `HF_HOME` | `~/.cache/huggingface` | HF cache directory |

To redirect the 90 GB model to a larger volume:

```bash
export HF_HOME=/workspace/.hf_home
mkdir -p "$HF_HOME"
./uniserve-fasth3-easy-install.sh
```

---

## What happens during install

The script runs seven steps in order. Each is prefixed with a `STEP N` banner and prints every subprocess command before running it.

| Step | Duration | What it does |
|---|---|---|
| 0 | 5 s | Preflight: verify GPU, check disk |
| 1 | 5 min | Install CUDA 13.0 toolkit via NVIDIA's apt repo |
| 2 | 2 min | Install `uv` and Rust toolchain |
| 3 | 10–20 min | Clone UniServe, build with `UV_TORCH_BACKEND=cu130` |
| 4 | 15–45 min | Download FastH3 NVFP4 (~90 GB, 68 files) |
| 5 | 1 s | Locate the single-GPU worker config |
| 6 | 0 s | Print serve instructions |
| 7 | 10–15 min | (Only with `--serve`) Launch server; first startup includes kernel compilation |

**Total: ~1 hour** for a fresh install on a good network.

---

## After install

### Start the server

```bash
cd /workspace/UniServe
source .venv/bin/activate

SNAPSHOT=$(find ~/.cache/huggingface/hub/models--FastVideo--FastVideo-FastH3-8-Step-V2-NVFP4/snapshots -maxdepth 1 -mindepth 1 -type d | head -1)

uniserve serve "$SNAPSHOT" \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-running-requests 1
```

The first launch compiles CUDA kernels and prepares CUDA graphs. Watch the log for:

```
worker ready
listening on 0.0.0.0:8000
```

### Verify

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

### Generate a test video

```bash
curl --fail-with-body --max-time 600 \
  http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "FastH3",
    "prompt": "CCTV footage of a woman in a supermarket aisle",
    "task": "t2va",
    "target": {
      "short_edge": 544,
      "aspect_ratio": "16:9",
      "duration_seconds": 5
    },
    "seed": 1000
  }' \
  --output test_video.mp4
```

This produces a **960×544, 5-second clip**. On an RTX 5090, expect **~20–30 s**.

---

## API reference

UniServe exposes an OpenAI-compatible video API:

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/models` | GET | List available models |
| `/v1/videos` | POST | Async job — returns a `video_id` immediately |
| `/v1/videos/{id}` | GET | Poll job status |
| `/v1/videos/{id}/content` | GET | Download the MP4 when complete |
| `/v1/videos/sync` | POST | Synchronous — blocks and returns the MP4 directly |
| `/health` | GET | Readiness probe |

### Sync request body

```json
{
  "model": "FastH3",
  "prompt": "your prompt here",
  "task": "t2va",
  "target": {
    "short_edge": 544,
    "aspect_ratio": "16:9",
    "duration_seconds": 5
  },
  "seed": 1000
}
```

| Field | Values | Notes |
|---|---|---|
| `task` | `t2va`, `i2va`, `fl2va` | text-to-video+audio, image-to-video+audio, first-last-frame-to-video |
| `target.short_edge` | 480, 544, 720 | Frame height (or width if portrait) |
| `target.aspect_ratio` | `16:9`, `9:16`, `1:1` | Combined with `short_edge` to pick resolution |
| `target.duration_seconds` | 4, 5, 8, 10 | Rounded to nearest valid frame count (`≡ 5 mod 17`) |
| `seed` | int | Deterministic across runs |

### Common resolutions

| `short_edge` | `aspect_ratio` | Output |
|---|---|---|
| 544 | `16:9` | 960×544 |
| 544 | `9:16` | 544×960 |
| 480 | `16:9` | 832×480 |
| 720 | `16:9` | 1280×720 |

---

## Adapting an existing batch script

If you have a script that talks to Unsloth Studio's `/api/inference/video/generate`, three endpoint substitutions convert it:

| Old (Unsloth Studio) | New (UniServe) |
|---|---|
| `POST /api/inference/video/generate` | `POST /v1/videos/sync` |
| `GET /api/inference/video/generate-progress` | *(not needed — sync blocks)* |
| `GET /api/inference/video/gallery/{id}/file` | *(sync returns MP4 directly)* |

Because the sync endpoint returns raw video bytes in the response body, your batch loop can drop the polling logic entirely — one POST per clip, write the response to disk, move on.

---

## Troubleshooting

### `CUDA_MISMATCH: The detected CUDA version (12.9) mismatches PyTorch (13.0)`

The system `nvcc` is older than what PyTorch was built against. Rerun the installer **without** `--skip-cuda` and verify:

```bash
nvcc --version                  # must report 13.0
python -c "import torch; print(torch.version.cuda)"   # must report 13.0
```

If `nvcc` still shows 12.x, the CUDA 13 binary isn't first in `PATH`:

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

### `_install_cuda13: command not found`

You're on the old version of the script. Redownload — the current version has no forward-referenced functions.

### HF download stalls or fails mid-transfer

Cancel and rerun — the HF CLI resumes from where it stopped:

```bash
hf download FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4
```

If it repeatedly stalls, disable Xet:

```bash
export HF_HUB_DISABLE_XET=1
hf download FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4
```

### Disk fills up during download

Check where the cache is landing:

```bash
df -h ~/.cache/huggingface
```

If it's a small root volume, redirect:

```bash
export HF_HOME=/workspace/.hf_home
mkdir -p "$HF_HOME"
# Then rerun the installer
```

### Server startup hangs at "worker startup"

The first launch compiles CUDA kernels and warms up the model — this takes **10–15 minutes** on a fresh install. Watch for these lines in the log:

```
worker still busy during Worker startup (NNN seconds)
```

These are informational, not errors. The server is ready when you see:

```
listening on 0.0.0.0:8000
```

### Out of memory during inference

The FastH3 NVFP4 model fits on a 32 GB RTX 5090, but barely. If you see OOM:

1. Set `--max-running-requests 1` (already the default in the install script)
2. Reduce `target.short_edge` to 480
3. Reduce `duration_seconds` to 4

For lower-residency loading on smaller GPUs:

```bash
uniserve serve "$SNAPSHOT" \
  --quantization-config '{"mode":"minimum"}' \
  ...
```

### Inference is slower than expected

- Check GPU clocks: `nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv`
- Confirm nothing else is using the GPU: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
- Confirm you're using the NVFP4 checkpoint, not a BF16 fallback. Look for `nvfp4` in the server log at model-load time.

---

## Performance reference

Measured on RTX 5090 (32 GB), FastH3 NVFP4, 5-second clips:

| Resolution | Time per clip |
|---|---|
| 832×480 | **14.8 s** |
| 960×544 | ~22 s |
| 1280×720 | **38.6 s** |

For 8,000 supermarket clips at 960×544: **~50 hours** of continuous GPU time.

---

## Uninstall

```bash
# Stop the server if running
pkill -f "uniserve serve"

# Remove UniServe source and venv
rm -rf /workspace/UniServe

# Remove the model cache (~90 GB)
rm -rf ~/.cache/huggingface/hub/models--FastVideo--FastVideo-FastH3-8-Step-V2-NVFP4

# Remove CUDA 13 (only if not needed for other workloads)
sudo apt-get remove -y 'cuda-toolkit-13-0' 'cuda-*13-0*'
sudo rm -f /etc/apt/preferences.d/cuda-repository-pin-600
sudo rm -rf /var/cuda-repo-ubuntu2404-13-0-local

# Remove uv and rust (optional)
rm -rf ~/.local/bin/uv ~/.local/bin/uvx
rustup self uninstall
```

---

## Further reading

- UniServe: https://github.com/hao-ai-lab/UniServe
- FastH3 model: https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4
- FastH3 benchmarks: https://hao-ai-lab.github.io/FastVideo/blog/fasth3_consumer_hardware.html

---

## License

The installer script is provided as-is. UniServe, FastH3, and their dependencies carry their own licenses — see the upstream repositories.
