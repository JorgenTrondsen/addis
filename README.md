# Cadis

## Prerequisites

Before installing, ensure your systems meets the following requirements:
- **OS:** Linux or WSL
- **Python:** Version 3.10 or higher
- **CUDA:** 12.2
- **NVIDIA Driver:** 535 or higher
- **Overlay Network Cluster** A populated cluster of devices using either Tailscale or Netbird

## Setup

Currently, Cadis relies on specific changes to the underlying inference engine. You will need to clone this repository alongside a forked repository of your desired inference engine.

*(Note: We currently have pending pull requests for `sglang` and `vLLM`. If they are not accepted, we will introduce these changes as monkey patches directly within Cadis in the future.)*

### 1. Clone Repositories

Depending on your engine of choice, clone the relevant fork so it sits in the same parent directory as `cadis`:

**For vLLM:**
```bash
git clone https://github.com/JorgenTrondsen/vllm.git
```

**For SGLang:**
```bash
git clone -b pp_optimization https://github.com/JorgenTrondsen/sglang.git
```

Your folder structure must look like this:
```text
person@machine:~$ ls
cadis  sglang  vllm
```

### 2. Install Build Tools

Upgrade `pip` and install `uv` (our recommended fast Python package installer):
```bash
pip install --upgrade pip
pip install uv
```

### 3. Install Your Preferred Inference Engine

Follow the steps below for the engine you intend to use.

#### Option A: SGLang Setup
```bash
cd sglang
uv venv venv
source venv/bin/activate
uv pip install -e "python"
```

#### Option B: vLLM Setup
```bash
cd vllm
uv venv venv
source venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --prerelease=allow --editable .
```

## Usage

### 1. Launching Cadis (Master)

To start the master node, which coordinates network profiling and pipeline calculation:

```bash
python src/launcher.py master \
    --num-workers 2 \
    --model-path /path/to/your/model \
    --inference-engine sglang \
    --gpu-memory-utilization 0.8 \
    --pp-async-batch-depth 2
```

**Common Arguments:**
- `--num-workers`: Number of worker nodes to wait for.
- `--model-path`: Path to the LLM weights.
- `--inference-engine`: Engine to use (`sglang` or `vllm`).
- `--gpu-memory-utilization`: Fraction of GPU memory to reserve (default: 0.8).
- `--pp-async-batch-depth`: Pipeline async batch depth for SGLang (default: 2).
- `--kv-cache-size`: KV cache memory size (default: 8G).
- `--overlay-network`: Overlay network type (`tailscale` or `netbird`).

### 2. Launching Cadis (Worker)

On each worker node, join the cluster by pointing to the master's IP:

```bash
python src/launcher.py worker <MASTER_IP>
```
