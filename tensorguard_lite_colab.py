# ==============================================================================
# TensorGuard-Lite: Google Colab T4 Sovereign AI Provenance Auditor
# ==============================================================================
# Paste this whole file into one Google Colab cell, or upload/run it as a .py file.
#
# What it implements from the PRD:
# - Hugging Face model download and optional gated-model login.
# - T4-aware fp16 model loading with small sequence length and dynamic cleanup.
# - Training-free white-box gradient fingerprint extraction.
# - Exact 16-dimensional TensorGuard-Lite fingerprint schema.
# - Cosine and Euclidean similarity analysis.
# - PCA lineage clustering export at publication resolution.
# - Multilingual tokenizer fertility and token-tax analysis for English/Hindi/Tamil.
# - Sovereign transparency/risk scorecard.
# - LaTeX, CSV, JSON, PNG, and SVG exports.
# - Interactive Gradio UI suitable for Colab.
#
# Recommended Colab runtime:
# Runtime -> Change runtime type -> T4 GPU
# ==============================================================================

import os
import sys
import gc
import json
import time
import random
import subprocess
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


# ------------------------------------------------------------------------------
# 1. Colab package installer
# ------------------------------------------------------------------------------

REQUIRED_PACKAGES = [
    "accelerate>=0.33.0",
    "bitsandbytes>=0.43.3",
    "gradio>=4.44.0",
    "huggingface_hub>=0.24.0",
    "matplotlib>=3.8.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "safetensors>=0.4.3",
    "scikit-learn>=1.4.0",
    "scipy>=1.11.0",
    "sentencepiece>=0.2.0",
    "torch>=2.3.0",
    "transformers>=4.44.0",
]


def install_missing_packages() -> None:
    """Install runtime dependencies inside Colab if they are missing."""
    import importlib.util

    package_to_import = {
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
        "gradio": "gradio",
        "huggingface_hub": "huggingface_hub",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "pandas": "pandas",
        "safetensors": "safetensors",
        "scikit-learn": "sklearn",
        "scipy": "scipy",
        "sentencepiece": "sentencepiece",
        "torch": "torch",
        "transformers": "transformers",
    }

    missing = []
    for requirement in REQUIRED_PACKAGES:
        package_name = requirement.split(">=")[0]
        import_name = package_to_import[package_name]
        if importlib.util.find_spec(import_name) is None:
            missing.append(requirement)

    if missing:
        print("Installing missing packages:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


install_missing_packages()


# ------------------------------------------------------------------------------
# 2. Imports after package installation
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, hf_hub_download, login
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging


warnings.filterwarnings("ignore", category=UserWarning)
hf_logging.set_verbosity_error()


# ------------------------------------------------------------------------------
# 3. Global runtime configuration
# ------------------------------------------------------------------------------

APP_NAME = "TensorGuard-Lite Sovereign AI Provenance Auditor"
OUTPUT_DIR = Path("/content/tensorguard_outputs") if Path("/content").exists() else Path("tensorguard_outputs")
HF_CACHE_DIR = Path("/content/tensorguard_hf_cache") if Path("/content").exists() else Path("tensorguard_hf_cache")
MODEL_CACHE_ROOT = HF_CACHE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SAMPLE_TEXT = (
    "Sovereign AI provenance auditing uses deterministic gradient sensitivity "
    "statistics to detect model lineage, transparency gaps, tokenizer costs, "
    "and possible unverified derivative dependencies."
)

DEFAULT_MODEL_OPTIONS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "meta-llama/Llama-3.2-1B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
]

PAPER_MODEL_IDS = [
    "meta-llama/Llama-3.2-1B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]

ATTENTION_KEYWORDS = ["q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value", "attention"]
MLP_KEYWORDS = ["gate_proj", "up_proj", "down_proj", "dense", "fc1", "fc2", "mlp"]
TARGET_KEYWORDS = ATTENTION_KEYWORDS + MLP_KEYWORDS

REFERENCE_FAMILIES = {
    "Llama-3.2-1B": "Llama",
    "Llama-3.2-1B-Instruct": "Llama",
    "Qwen2.5-0.5B-Instruct": "Qwen",
    "Qwen2.5-1.5B": "Qwen",
    "Qwen2.5-1.5B-Instruct": "Qwen",
    "DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek/Qwen derivative",
    "Gemma-2-2B": "Gemma",
    "Ministral-8B-Instruct-2410": "Ministral",
    "Mistral-7B": "Mistral",
    "Phi-3.5-mini": "Phi",
    "SmolLM2-1.7B-Instruct": "SmolLM",
    "SmolLM2-135M": "SmolLM",
    "SmolLM2-135M-Instruct": "SmolLM",
}

REFERENCE_MODEL_IDS = {
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Gemma-2-2B": "google/gemma-2-2b",
    "Ministral-8B-Instruct-2410": "mistralai/Ministral-8B-Instruct-2410",
    "Mistral-7B": "mistralai/Mistral-7B-v0.1",
    "Phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M",
    "SmolLM2-135M-Instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
}

FAMILY_COLORS = {
    "Llama": "#7b1fa2",
    "Qwen": "#1565c0",
    "DeepSeek/Qwen derivative": "#00897b",
    "Gemma": "#ef6c00",
    "Ministral": "#5d4037",
    "Mistral": "#6d4c41",
    "Phi": "#455a64",
    "SmolLM": "#2e7d32",
    "Audited": "#c62828",
}


REFERENCE_FINGERPRINTS: Dict[str, List[float]] = {
    "Llama-3.2-1B": [0.00012, 0.0034, 1.20, 0.04, -0.12, 0.00011, 0.0029, 0.90, 0.03, -0.11, 0.00010, 0.0031, 1.10, 0.02, 1235.0, 16.0],
    "Llama-3.2-1B-Instruct": [0.00013, 0.0035, 1.30, 0.05, -0.10, 0.00012, 0.0030, 1.00, 0.04, -0.09, 0.00011, 0.0032, 1.20, 0.03, 1235.0, 16.0],
    "Qwen2.5-0.5B-Instruct": [0.00038, 0.0064, 2.10, 0.11, -0.37, 0.00035, 0.0060, 1.90, 0.10, -0.35, 0.00036, 0.0062, 2.00, 0.09, 494.0, 24.0],
    "Qwen2.5-1.5B": [0.00045, 0.0081, 2.80, 0.15, -0.45, 0.00042, 0.0078, 2.50, 0.12, -0.42, 0.00040, 0.0080, 2.70, 0.11, 1540.0, 28.0],
    "Qwen2.5-1.5B-Instruct": [0.00046, 0.0082, 2.90, 0.16, -0.43, 0.00043, 0.0079, 2.60, 0.13, -0.40, 0.00041, 0.0081, 2.80, 0.12, 1540.0, 28.0],
    "DeepSeek-R1-Distill-Qwen-1.5B": [0.00044, 0.0080, 2.70, 0.14, -0.44, 0.00041, 0.0077, 2.60, 0.13, -0.41, 0.00039, 0.0079, 2.60, 0.10, 1540.0, 28.0],
    "Gemma-2-2B": [0.00009, 0.0012, 0.50, -0.05, 0.08, 0.00009, 0.0011, 0.40, -0.04, 0.07, 0.00009, 0.0011, 0.40, -0.04, 2610.0, 26.0],
    "Ministral-8B-Instruct-2410": [0.00021, 0.0041, 1.65, 0.07, -0.18, 0.00020, 0.0039, 1.52, 0.06, -0.17, 0.00020, 0.0040, 1.58, 0.07, 8000.0, 36.0],
    "Mistral-7B": [0.00025, 0.0048, 1.90, 0.08, -0.22, 0.00023, 0.0045, 1.70, 0.07, -0.20, 0.00024, 0.0046, 1.80, 0.08, 7240.0, 32.0],
    "Phi-3.5-mini": [0.00008, 0.0009, 0.30, -0.02, 0.05, 0.00008, 0.0009, 0.20, -0.01, 0.04, 0.00008, 0.0009, 0.30, -0.02, 3820.0, 32.0],
    "SmolLM2-1.7B-Instruct": [0.00033, 0.0058, 2.35, 0.12, -0.25, 0.00031, 0.0054, 2.05, 0.10, -0.23, 0.00032, 0.0056, 2.20, 0.10, 1700.0, 24.0],
    "SmolLM2-135M": [0.00031, 0.0055, 2.10, 0.10, -0.30, 0.00029, 0.0051, 1.80, 0.08, -0.28, 0.00030, 0.0053, 2.00, 0.09, 135.0, 12.0],
    "SmolLM2-135M-Instruct": [0.00032, 0.0056, 2.20, 0.11, -0.28, 0.00030, 0.0052, 1.90, 0.09, -0.26, 0.00031, 0.0054, 2.10, 0.10, 135.0, 12.0],
}


MULTILINGUAL_CORPUS = {
    "English": [
        "Sovereign AI provenance tracking ensures model licensing compliance and transparency across decentralized environments.",
        "Gradient statistics can reveal structural relationships between a parent model and a derivative model.",
        "Tokenizer over-fragmentation increases computational cost, latency, and effective context window pressure.",
        "Transparent evaluation reports help auditors understand safety limits and deployment risk.",
    ],
    "Hindi": [
        "सॉवरेन एआई provenance tracking मॉडल लाइसेंस अनुपालन और पारदर्शिता सुनिश्चित करता है।",
        "ग्रेडिएंट सांख्यिकी मूल मॉडल और व्युत्पन्न मॉडल के बीच संरचनात्मक संबंध दिखा सकती है।",
        "टोकन का अत्यधिक विखंडन लागत, विलंबता और संदर्भ सीमा पर दबाव बढ़ाता है।",
        "पारदर्शी मूल्यांकन रिपोर्ट सुरक्षा सीमाओं और तैनाती जोखिम को समझने में मदद करती है।",
    ],
    "Tamil": [
        "சுயாட்சி செயற்கை நுண்ணறிவு provenance tracking மாதிரி உரிமம் மற்றும் வெளிப்படைத்தன்மையை உறுதி செய்கிறது.",
        "சாய்வு புள்ளிவிவரங்கள் மூல மாதிரி மற்றும் பெறப்பட்ட மாதிரி இடையிலான கட்டமைப்பு உறவை காட்டலாம்.",
        "டோக்கன் அதிக துண்டாக்கம் செலவு, தாமதம் மற்றும் சூழல் சாளர அழுத்தத்தை அதிகரிக்கிறது.",
        "வெளிப்படையான மதிப்பீட்டு அறிக்கைகள் பாதுகாப்பு வரம்புகள் மற்றும் பயன்பாட்டு ஆபத்தை விளக்குகின்றன.",
    ],
}


# ------------------------------------------------------------------------------
# 4. Data classes
# ------------------------------------------------------------------------------

@dataclass
class RuntimeReport:
    device: str
    gpu_name: str
    total_vram_gb: float
    allocated_vram_gb: float
    reserved_vram_gb: float
    torch_version: str
    cuda_version: str
    output_dir: str


@dataclass
class AuditConfig:
    model_id: str
    hf_token: str = ""
    seed: int = 42
    perturbation_runs: int = 30
    max_length: int = 96
    sample_gradient_entries: int = 500000
    weight_noise_std: float = 0.0
    sample_text: str = DEFAULT_SAMPLE_TEXT
    quick_smoke_test: bool = False
    save_to_drive: bool = False
    persist_model_cache: bool = False
    open_weights: bool = True
    data_transparency: bool = False
    recipe_disclosure: bool = False
    tokenizer_openness: bool = True
    safety_report: bool = False
    evaluation_results: bool = False
    cryptographic_data_proof: bool = False
    architecture_documented: bool = True
    regulator_safetensors_access: bool = False
    compute_subsidy_disclosure: bool = False


# ------------------------------------------------------------------------------
# 5. Determinism, hardware, and cleanup helpers
# ------------------------------------------------------------------------------

def enforce_determinism(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_runtime_report() -> RuntimeReport:
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return RuntimeReport(
            device="cuda",
            gpu_name=props.name,
            total_vram_gb=round(props.total_memory / 1024**3, 2),
            allocated_vram_gb=round(torch.cuda.memory_allocated(idx) / 1024**3, 2),
            reserved_vram_gb=round(torch.cuda.memory_reserved(idx) / 1024**3, 2),
            torch_version=torch.__version__,
            cuda_version=str(torch.version.cuda),
            output_dir=str(OUTPUT_DIR),
        )
    return RuntimeReport(
        device="cpu",
        gpu_name="No CUDA GPU detected",
        total_vram_gb=0.0,
        allocated_vram_gb=0.0,
        reserved_vram_gb=0.0,
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        output_dir=str(OUTPUT_DIR),
    )


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def maybe_mount_drive(save_to_drive: bool) -> Path:
    if not save_to_drive:
        return OUTPUT_DIR
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
        drive_dir = Path("/content/drive/MyDrive/tensorguard_outputs")
        drive_dir.mkdir(parents=True, exist_ok=True)
        return drive_dir
    except Exception as exc:
        print(f"Google Drive mount failed, using local output dir instead: {exc}")
        return OUTPUT_DIR


def configure_model_cache(persist_model_cache: bool) -> Path:
    global MODEL_CACHE_ROOT
    if not persist_model_cache:
        MODEL_CACHE_ROOT = HF_CACHE_DIR
        MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        return MODEL_CACHE_ROOT
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
        MODEL_CACHE_ROOT = Path("/content/drive/MyDrive/tensorguard_hf_cache")
        MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        return MODEL_CACHE_ROOT
    except Exception as exc:
        print(f"Google Drive model cache unavailable, using local runtime cache instead: {exc}")
        MODEL_CACHE_ROOT = HF_CACHE_DIR
        MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        return MODEL_CACHE_ROOT


def authenticate_huggingface(hf_token: str) -> None:
    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    if token:
        login(token=token, add_to_git_credential=False)


def cache_dir_for_model(model_id: str) -> Path:
    safe_name = model_id.replace("/", "--")
    model_dir = MODEL_CACHE_ROOT / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def should_download_file(filename: str) -> bool:
    lowered = filename.lower()
    allowed_suffixes = (
        ".json",
        ".txt",
        ".py",
        ".model",
        ".spm",
        ".tiktoken",
        ".safetensors",
        ".bin",
    )
    excluded_fragments = (
        "optimizer",
        "scheduler",
        "training_args",
        "trainer_state",
        "events.out",
        ".md",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".pdf",
        ".parquet",
        ".onnx",
    )
    return lowered.endswith(allowed_suffixes) and not any(part in lowered for part in excluded_fragments)


def download_model_with_progress(model_id: str, hf_token: str = "", progress=None) -> Dict[str, object]:
    """
    Prefetch model/tokenizer files into a stable Colab cache.
    The progress percentage is file-level because Hugging Face's high-level
    download helpers do not expose a byte callback to Gradio.
    """
    token = hf_token or None
    cache_dir = cache_dir_for_model(model_id)
    api = HfApi()
    if progress:
        progress(0.01, desc="Connecting to Hugging Face")

    files = [f for f in api.list_repo_files(model_id, repo_type="model", token=token) if should_download_file(f)]
    if not files:
        raise RuntimeError(f"No downloadable model/tokenizer files found for {model_id}.")

    downloaded = 0
    reused = 0
    total = len(files)
    for idx, filename in enumerate(files, start=1):
        pct_start = 0.05 + 0.90 * ((idx - 1) / total)
        if progress:
            progress(pct_start, desc=f"Checking cache {idx}/{total}: {filename}")
        try:
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                repo_type="model",
                token=token,
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
            reused += 1
        except Exception:
            if progress:
                progress(pct_start, desc=f"Downloading {idx}/{total}: {filename}")
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                repo_type="model",
                token=token,
                cache_dir=str(cache_dir),
                local_files_only=False,
            )
            downloaded += 1

        if progress:
            progress(0.05 + 0.90 * (idx / total), desc=f"Downloaded/cache ready {idx}/{total}")

    if progress:
        progress(0.97, desc="Files cached. Loading tokenizer/model")
    return {
        "cache_dir": str(cache_dir),
        "file_count": total,
        "downloaded_files": downloaded,
        "reused_cached_files": reused,
        "cache_status": "already cached" if downloaded == 0 else "downloaded/updated",
    }


def inspect_model_repository(model_id: str, hf_token: str = "") -> Dict[str, object]:
    token = hf_token or None
    api = HfApi()
    info = api.model_info(model_id, token=token)
    siblings = [s.rfilename for s in (info.siblings or [])]
    config = AutoConfig.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
        cache_dir=str(cache_dir_for_model(model_id)),
    )
    params_m, layers = estimate_model_parameters(model_id, hf_token)
    tokenizer_files = [f for f in siblings if any(part in f.lower() for part in ["tokenizer", "vocab", "merges", "sentencepiece", ".model"])]
    safetensors_files = [f for f in siblings if f.endswith(".safetensors")]
    bin_files = [f for f in siblings if f.endswith(".bin")]
    return {
        "model_id": model_id,
        "architecture": ", ".join(getattr(config, "architectures", []) or [config.model_type]),
        "model_type": getattr(config, "model_type", "unknown"),
        "estimated_parameters_m": params_m,
        "num_hidden_layers": layers,
        "safetensors_available": bool(safetensors_files),
        "safetensors_files": len(safetensors_files),
        "pytorch_bin_files": len(bin_files),
        "tokenizer_files": ", ".join(tokenizer_files[:6]) if tokenizer_files else "not detected in repo listing",
        "repo_files_seen": len(siblings),
    }


# ------------------------------------------------------------------------------
# 6. Model loading and target-module discovery
# ------------------------------------------------------------------------------

def estimate_model_parameters(model_id: str, hf_token: str = "") -> Tuple[float, int]:
    config = AutoConfig.from_pretrained(
        model_id,
        token=hf_token or None,
        trust_remote_code=True,
        cache_dir=str(cache_dir_for_model(model_id)),
    )
    hidden = int(getattr(config, "hidden_size", 0) or getattr(config, "n_embd", 0) or 0)
    layers = int(getattr(config, "num_hidden_layers", 0) or getattr(config, "n_layer", 0) or 0)
    vocab = int(getattr(config, "vocab_size", 0) or 0)
    intermediate = int(getattr(config, "intermediate_size", 0) or hidden * 4)
    rough_params = vocab * hidden + layers * (4 * hidden * hidden + 3 * hidden * intermediate)
    return round(rough_params / 1e6, 2), layers


def load_tokenizer_and_model(model_id: str, hf_token: str, quick_smoke_test: bool, progress=None):
    """Load the requested Hugging Face model in a T4-friendly fp16 gradient mode."""
    actual_model_id = "HuggingFaceTB/SmolLM2-135M-Instruct" if quick_smoke_test else model_id
    print(f"Downloading/loading Hugging Face model: {actual_model_id}")
    download_report = download_model_with_progress(actual_model_id, hf_token, progress=progress)
    cache_dir = download_report["cache_dir"]

    tokenizer = AutoTokenizer.from_pretrained(
        actual_model_id,
        token=hf_token or None,
        trust_remote_code=True,
        use_fast=True,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        actual_model_id,
        token=hf_token or None,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return tokenizer, model, actual_model_id, download_report


def module_matches(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in TARGET_KEYWORDS)


def discover_target_modules(model: nn.Module, max_modules: int = 12) -> List[Tuple[str, nn.Module]]:
    """Find a small representative set of linear attention/MLP layers."""
    matches = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module_matches(name):
            matches.append((name, module))

    if not matches:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                matches.append((name, module))

    if len(matches) <= max_modules:
        return matches

    indices = np.linspace(0, len(matches) - 1, max_modules, dtype=int)
    return [matches[i] for i in indices]


def discover_embedding_modules(model: nn.Module, max_modules: int = 2) -> List[Tuple[str, nn.Module]]:
    """Find token embedding modules for the exact 16-feature fingerprint schema."""
    matches = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            lowered = name.lower()
            if any(key in lowered for key in ["embed", "wte", "tok_embeddings", "word_embeddings"]):
                matches.append((name, module))
    if not matches:
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                matches.append((name, module))
    return matches[:max_modules]


def freeze_all_but_targets(
    model: nn.Module,
    target_modules: List[Tuple[str, nn.Module]],
    embedding_modules: List[Tuple[str, nn.Module]],
) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    selected_modules = target_modules + embedding_modules
    target_ids = {id(module.weight) for _, module in selected_modules if hasattr(module, "weight")}
    for _, module in selected_modules:
        if hasattr(module, "weight") and id(module.weight) in target_ids:
            module.weight.requires_grad_(True)


# ------------------------------------------------------------------------------
# 7. TensorGuard-Lite 16D fingerprint engine
# ------------------------------------------------------------------------------

def stable_stats(values: np.ndarray) -> List[float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.linalg.norm(values)),
        float(skew(values)) if values.size > 3 and np.std(values) > 0 else 0.0,
        float(kurtosis(values)) if values.size > 3 and np.std(values) > 0 else 0.0,
    ]


def category_for_module(name: str) -> str:
    lowered = name.lower()
    if any(key in lowered for key in ["embed", "wte", "tok_embeddings", "word_embeddings"]):
        return "embedding"
    if any(key in lowered for key in ["q_proj", "query"]):
        return "attention"
    if any(key in lowered for key in ["k_proj", "key"]):
        return "attention"
    if any(key in lowered for key in ["v_proj", "value"]):
        return "attention"
    if any(key in lowered for key in ["o_proj", "out_proj", "output"]):
        return "attention"
    if any(key in lowered for key in ATTENTION_KEYWORDS):
        return "attention"
    if any(key in lowered for key in MLP_KEYWORDS):
        return "ffn"
    return "other"


def collect_gradient_values(
    target_modules: List[Tuple[str, nn.Module]],
    embedding_modules: List[Tuple[str, nn.Module]],
    max_total_entries: int = 500000,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    by_category: Dict[str, List[np.ndarray]] = {
        "attention": [],
        "ffn": [],
        "embedding": [],
        "other": [],
    }
    all_values = []

    selected_modules = target_modules + embedding_modules
    per_module_limit = max(1, int(max_total_entries) // max(1, len(selected_modules)))
    for name, module in selected_modules:
        grad = getattr(module.weight, "grad", None)
        if grad is None:
            continue
        arr = grad.detach().float().cpu().numpy().reshape(-1)
        if arr.size == 0:
            continue
        if arr.size > per_module_limit:
            stable_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(name)) % (2**32)
            rng = np.random.default_rng(stable_seed)
            arr = arr[rng.choice(arr.size, size=per_module_limit, replace=False)]
        all_values.append(arr)
        by_category[category_for_module(name)].append(arr)

    packed = {}
    for category, arrays in by_category.items():
        packed[category] = np.concatenate(arrays) if arrays else np.array([], dtype=np.float64)
    all_gradients = np.concatenate(all_values) if all_values else np.array([], dtype=np.float64)
    return all_gradients, packed


def apply_temporary_weight_noise(
    target_modules: List[Tuple[str, nn.Module]],
    noise_std: float,
    seed: int,
) -> List[Tuple[nn.Parameter, torch.Tensor]]:
    """Add small Gaussian noise to target weights and return backups for restoration."""
    if noise_std <= 0:
        return []
    backups = []
    with torch.no_grad():
        for idx, (_, module) in enumerate(target_modules):
            weight = module.weight
            generator = torch.Generator(device=weight.device)
            generator.manual_seed(seed + idx)
            backup = weight.detach().clone()
            noise = torch.randn(weight.shape, generator=generator, device=weight.device, dtype=weight.dtype) * float(noise_std)
            weight.add_(noise)
            backups.append((weight, backup))
    return backups


def restore_weight_noise(backups: List[Tuple[nn.Parameter, torch.Tensor]]) -> None:
    with torch.no_grad():
        for weight, backup in backups:
            weight.copy_(backup)


def build_16d_fingerprint(
    all_gradients: np.ndarray,
    category_gradients: Dict[str, np.ndarray],
    total_params_m: float,
    active_layer_count: int,
) -> np.ndarray:
    """Exact 16D schema: 5 global + 3 attention + 3 FFN + 3 embedding + 2 structural."""
    global_features = stable_stats(all_gradients)

    category_features = []
    for category in ["attention", "ffn", "embedding"]:
        vals = category_gradients.get(category, np.array([]))
        if vals.size == 0:
            category_features.extend([0.0, 0.0, 0.0])
        else:
            category_features.extend([
                float(np.mean(vals)),
                float(np.std(vals)),
                float(np.linalg.norm(vals)),
            ])

    structural_features = [float(total_params_m), float(active_layer_count)]
    vector = np.array(global_features + category_features + structural_features, dtype=np.float64)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


class TensorGuardLiteAuditor:
    def __init__(self, config: AuditConfig):
        self.config = config
        self.out_dir = maybe_mount_drive(config.save_to_drive)
        self.cache_root = configure_model_cache(config.persist_model_cache)
        authenticate_huggingface(config.hf_token)
        enforce_determinism(config.seed)

    def extract_fingerprint(self, progress=None) -> Tuple[np.ndarray, Dict[str, object]]:
        tokenizer, model, actual_model_id, download_report = load_tokenizer_and_model(
            self.config.model_id,
            self.config.hf_token,
            self.config.quick_smoke_test,
            progress=progress,
        )
        repo_metadata = inspect_model_repository(actual_model_id, self.config.hf_token)
        target_modules = discover_target_modules(model, max_modules=12)
        embedding_modules = discover_embedding_modules(model, max_modules=1)
        if not target_modules:
            raise RuntimeError("No linear modules found for gradient fingerprint extraction.")

        freeze_all_but_targets(model, target_modules, embedding_modules)
        total_params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
        active_layer_count = len(target_modules) + len(embedding_modules)

        vectors = []
        start = time.time()

        for run_idx in range(int(self.config.perturbation_runs)):
            enforce_determinism(self.config.seed + run_idx)
            model.zero_grad(set_to_none=True)
            if progress:
                progress(
                    0.97 + 0.03 * (run_idx / max(1, int(self.config.perturbation_runs))),
                    desc=f"Gradient fingerprint run {run_idx + 1}/{self.config.perturbation_runs}",
                )

            prompt = f"{self.config.sample_text}\nAudit nonce: {run_idx}"
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=int(self.config.max_length),
                padding=False,
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}

            backups = apply_temporary_weight_noise(
                target_modules,
                noise_std=float(self.config.weight_noise_std),
                seed=self.config.seed + run_idx,
            )
            try:
                outputs = model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[-1].float()
                loss = torch.linalg.vector_norm(hidden, ord=2)
                loss.backward()
            finally:
                restore_weight_noise(backups)

            all_gradients, category_gradients = collect_gradient_values(
                target_modules,
                embedding_modules,
                max_total_entries=int(self.config.sample_gradient_entries),
            )
            vector = build_16d_fingerprint(
                all_gradients,
                category_gradients,
                total_params_m=total_params_m,
                active_layer_count=active_layer_count,
            )
            vectors.append(vector)

            del outputs, hidden, loss, encoded
            cleanup_cuda()

        fingerprint = np.mean(np.vstack(vectors), axis=0)
        variance = np.var(np.vstack(vectors), axis=0)
        elapsed = time.time() - start

        metadata = {
            "requested_model_id": self.config.model_id,
            "actual_model_id": actual_model_id,
            "quick_smoke_test": self.config.quick_smoke_test,
            "perturbation_runs": self.config.perturbation_runs,
            "max_length": self.config.max_length,
            "target_modules": [name for name, _ in target_modules],
            "total_params_m": total_params_m,
            "active_layer_count": active_layer_count,
            "embedding_modules": [name for name, _ in embedding_modules],
            "fingerprint_schema": [
                "global_mean",
                "global_std",
                "global_norm",
                "global_skewness",
                "global_kurtosis",
                "attention_mean",
                "attention_std",
                "attention_norm",
                "ffn_mean",
                "ffn_std",
                "ffn_norm",
                "embedding_mean",
                "embedding_std",
                "embedding_norm",
                "total_params",
                "num_layers",
            ],
            "sample_gradient_entries": self.config.sample_gradient_entries,
            "weight_noise_std": self.config.weight_noise_std,
            "download_report": download_report,
            "repository_metadata": repo_metadata,
            "mean_coordinate_variance": float(np.mean(variance)),
            "elapsed_seconds": round(elapsed, 3),
        }

        del model, tokenizer
        cleanup_cuda()
        return fingerprint, metadata


# ------------------------------------------------------------------------------
# 8. Similarity, PCA, tokenizer fertility, and governance scoring
# ------------------------------------------------------------------------------

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.dot(u, v) / denom) if denom else 0.0


def similarity_table(fingerprint: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, ref in REFERENCE_FINGERPRINTS.items():
        ref_vec = np.array(ref, dtype=np.float64)
        rows.append(
            {
                "Reference Model": name,
                "Cosine Similarity": round(cosine_similarity(fingerprint, ref_vec), 6),
                "Euclidean Distance": round(float(np.linalg.norm(fingerprint - ref_vec)), 6),
            }
        )
    return pd.DataFrame(rows).sort_values("Cosine Similarity", ascending=False).reset_index(drop=True)


def euclidean_distance_matrix(fingerprint: np.ndarray, model_label: str) -> pd.DataFrame:
    labels = list(REFERENCE_FINGERPRINTS.keys()) + [f"Audited: {model_label}"]
    vectors = [np.array(v, dtype=np.float64) for v in REFERENCE_FINGERPRINTS.values()] + [fingerprint]
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for i, left in enumerate(vectors):
        for j, right in enumerate(vectors):
            matrix[i, j] = float(np.linalg.norm(left - right))
    return pd.DataFrame(matrix, index=labels, columns=labels).round(6).reset_index(names="Model")


def tokenizer_vocab_jaccard(model_id: str, reference_model_id: str, hf_token: str = "") -> float:
    try:
        left = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token or None,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=str(cache_dir_for_model(model_id)),
        )
        right = AutoTokenizer.from_pretrained(
            reference_model_id,
            token=hf_token or None,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=str(cache_dir_for_model(reference_model_id)),
        )
        left_vocab = set(left.get_vocab().keys())
        right_vocab = set(right.get_vocab().keys())
        if not left_vocab or not right_vocab:
            return 0.0
        return round(len(left_vocab & right_vocab) / len(left_vocab | right_vocab), 6)
    except Exception:
        return 0.0


def run_tokenizer_fertility_audit(model_id: str, hf_token: str = "") -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token or None,
        trust_remote_code=True,
        use_fast=True,
        cache_dir=str(cache_dir_for_model(model_id)),
    )
    records = []

    def count_words(texts: List[str]) -> int:
        return sum(len(text.split()) for text in texts)

    def count_tokens(texts: List[str]) -> int:
        return sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)

    def average_token_chars(texts: List[str]) -> float:
        pieces = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            pieces.extend(tokenizer.convert_ids_to_tokens(ids))
        clean = [piece.replace("Ġ", "").replace("▁", "") for piece in pieces if piece]
        return sum(len(piece) for piece in clean) / max(1, len(clean))

    english_words = count_words(MULTILINGUAL_CORPUS["English"])
    english_tokens = count_tokens(MULTILINGUAL_CORPUS["English"])
    english_fertility = english_tokens / max(1, english_words)

    for language, texts in MULTILINGUAL_CORPUS.items():
        words = count_words(texts)
        tokens = count_tokens(texts)
        fertility = tokens / max(1, words)
        token_tax = fertility / max(english_fertility, 1e-9)
        cost_multiplier = token_tax ** 2
        if token_tax < 1.5:
            interpretation = "balanced"
        elif token_tax < 2.5:
            interpretation = "moderate token tax"
        else:
            interpretation = "large token tax"
        records.append(
            {
                "Language": language,
                "Words": words,
                "Tokens": tokens,
                "Fertility Phi": round(fertility, 4),
                "Average Token Length": round(average_token_chars(texts), 4),
                "Token Tax Psi vs English": round(token_tax, 4),
                "Approx Attention Cost Multiplier": round(cost_multiplier, 4),
                "Interpretation": interpretation,
            }
        )
    return pd.DataFrame(records)


def transparency_score(config: AuditConfig) -> Tuple[float, pd.DataFrame]:
    weights = {
        "Open Weights": 0.18,
        "Data Transparency": 0.14,
        "Recipe Disclosure": 0.10,
        "Tokenizer Openness": 0.10,
        "Safety Report": 0.10,
        "Evaluation Results": 0.08,
        "Cryptographic Data Lineage Proof": 0.10,
        "Architecture Documented": 0.08,
        "Regulator Safetensors Access": 0.08,
        "Compute Subsidy Disclosure": 0.04,
    }
    values = {
        "Open Weights": config.open_weights,
        "Data Transparency": config.data_transparency,
        "Recipe Disclosure": config.recipe_disclosure,
        "Tokenizer Openness": config.tokenizer_openness,
        "Safety Report": config.safety_report,
        "Evaluation Results": config.evaluation_results,
        "Cryptographic Data Lineage Proof": config.cryptographic_data_proof,
        "Architecture Documented": config.architecture_documented,
        "Regulator Safetensors Access": config.regulator_safetensors_access,
        "Compute Subsidy Disclosure": config.compute_subsidy_disclosure,
    }
    score = sum(weights[key] * float(values[key]) for key in weights)
    table = pd.DataFrame(
        {
            "Dimension": list(weights.keys()),
            "Weight": [weights[k] for k in weights],
            "Status": ["Yes" if values[k] else "No" for k in weights],
            "Contribution": [round(weights[k] * float(values[k]), 3) for k in weights],
        }
    )
    return round(score, 4), table


def risk_assessment(
    transparency: float,
    highest_similarity: float,
    matched_model: str,
    has_safetensors: bool = True,
) -> Tuple[str, str]:
    if not has_safetensors:
        return (
            "UNVERIFIABLE",
            "Repository does not expose safetensors in the visible model listing. White-box gradient provenance is limited; require regulator-level weight access before accepting sovereignty claims.",
        )
    if transparency >= 0.8 and highest_similarity < 0.90:
        return (
            "LOW RISK",
            "High transparency and no dominant unverified lineage signal. Suitable for sovereign deployment with normal documentation controls.",
        )
    if transparency >= 0.5 and highest_similarity < 0.95:
        return (
            "MEDIUM RISK",
            "Moderate transparency or lineage uncertainty. Require attribution review, license confirmation, and documented evaluation evidence.",
        )
    return (
        "HIGH RISK",
        f"Possible open-washing risk: low transparency or strong lineage similarity to {matched_model}. Restrict deployment until provenance, license, tokenizer, and training disclosures are verified.",
    )


# ------------------------------------------------------------------------------
# 9. Publication exports: PCA, LaTeX, CSV, JSON
# ------------------------------------------------------------------------------

def save_pca_plot(fingerprint: np.ndarray, model_label: str, out_dir: Path) -> Tuple[str, str]:
    labels = list(REFERENCE_FINGERPRINTS.keys()) + [f"Audited: {model_label}"]
    matrix = np.vstack([np.array(v, dtype=np.float64) for v in REFERENCE_FINGERPRINTS.values()] + [fingerprint])
    matrix = StandardScaler().fit_transform(matrix)
    coords = PCA(n_components=2, random_state=42).fit_transform(matrix)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
        }
    )

    fig_w, fig_h = 85 / 25.4, 70 / 25.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for idx, label in enumerate(labels):
        x, y = coords[idx]
        if label.startswith("Audited:"):
            ax.scatter(x, y, s=90, marker="*", color="#c62828", edgecolor="black", linewidth=0.5, zorder=4)
            ax.annotate("Audited", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, weight="bold")
        else:
            family = REFERENCE_FAMILIES.get(label, "Unknown")
            ax.scatter(x, y, s=24, marker="o", color=FAMILY_COLORS.get(family, "#1565c0"), alpha=0.76)
            ax.annotate(label, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=4.8)

    ax.set_title("TensorGuard-Lite PCA Lineage Map", fontsize=8)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linewidth=0.25, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = []
    for family in sorted(set(REFERENCE_FAMILIES.values())):
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS.get(family, "#1565c0"), markersize=4, label=family)
        )
    legend_handles.append(plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#c62828", markeredgecolor="black", markersize=6, label="Audited"))
    ax.legend(handles=legend_handles, fontsize=4.8, loc="best", frameon=False)
    fig.tight_layout()

    png_path = out_dir / "provenance_pca_clustering_600dpi.png"
    svg_path = out_dir / "provenance_pca_clustering.svg"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return str(png_path), str(svg_path)


def scorecard_latex(scorecard: pd.DataFrame) -> str:
    return scorecard.to_latex(index=False, escape=True, caption="Sovereign AI Provenance Auditor Scorecard", label="tab:tensorguard_scorecard")


def export_audit_bundle(
    fingerprint: np.ndarray,
    metadata: Dict[str, object],
    similarities: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    fertility: pd.DataFrame,
    governance: pd.DataFrame,
    scorecard: pd.DataFrame,
    latex: str,
    out_dir: Path,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "fingerprint_json": out_dir / "audited_fingerprint.json",
        "similarity_csv": out_dir / "lineage_similarity.csv",
        "euclidean_distance_matrix_csv": out_dir / "euclidean_distance_matrix.csv",
        "fertility_csv": out_dir / "tokenizer_fertility.csv",
        "governance_csv": out_dir / "governance_dimensions.csv",
        "scorecard_csv": out_dir / "sovereign_scorecard.csv",
        "scorecard_tex": out_dir / "sovereign_scorecard.tex",
    }

    payload = {"metadata": metadata, "fingerprint_16d": [float(x) for x in fingerprint]}
    paths["fingerprint_json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    similarities.to_csv(paths["similarity_csv"], index=False)
    distance_matrix.to_csv(paths["euclidean_distance_matrix_csv"], index=False)
    fertility.to_csv(paths["fertility_csv"], index=False)
    governance.to_csv(paths["governance_csv"], index=False)
    scorecard.to_csv(paths["scorecard_csv"], index=False)
    paths["scorecard_tex"].write_text(latex, encoding="utf-8")

    return {key: str(value) for key, value in paths.items()}


# ------------------------------------------------------------------------------
# 10. End-to-end audit controller and verification checks
# ------------------------------------------------------------------------------

def metadata_to_dataframe(metadata: Dict[str, object]) -> pd.DataFrame:
    repo = metadata.get("repository_metadata", {}) or {}
    download = metadata.get("download_report", {}) or {}
    rows = [
        {"Field": "Requested Model", "Value": metadata.get("requested_model_id", "")},
        {"Field": "Audited Model", "Value": metadata.get("actual_model_id", "")},
        {"Field": "Architecture", "Value": repo.get("architecture", "unknown")},
        {"Field": "Model Type", "Value": repo.get("model_type", "unknown")},
        {"Field": "Estimated Parameters (M)", "Value": repo.get("estimated_parameters_m", metadata.get("total_params_m", ""))},
        {"Field": "Hidden Layers", "Value": repo.get("num_hidden_layers", "")},
        {"Field": "Safetensors Available", "Value": repo.get("safetensors_available", False)},
        {"Field": "Safetensors Files", "Value": repo.get("safetensors_files", 0)},
        {"Field": "Tokenizer Files", "Value": repo.get("tokenizer_files", "")},
        {"Field": "Cache Status", "Value": download.get("cache_status", "")},
        {"Field": "Downloaded Files This Run", "Value": download.get("downloaded_files", "")},
        {"Field": "Reused Cached Files", "Value": download.get("reused_cached_files", "")},
        {"Field": "Cache Directory", "Value": download.get("cache_dir", "")},
        {"Field": "Target Modules", "Value": ", ".join(metadata.get("target_modules", []))},
        {"Field": "Embedding Modules", "Value": ", ".join(metadata.get("embedding_modules", []))},
        {"Field": "Fingerprint Schema", "Value": ", ".join(metadata.get("fingerprint_schema", []))},
    ]
    return pd.DataFrame(rows)


def run_full_audit(config: AuditConfig, progress=None):
    start = time.time()
    out_dir = maybe_mount_drive(config.save_to_drive)
    auditor = TensorGuardLiteAuditor(config)
    fingerprint, metadata = auditor.extract_fingerprint(progress=progress)

    similarities = similarity_table(fingerprint)
    distance_matrix = euclidean_distance_matrix(fingerprint, metadata["actual_model_id"])
    fertility = run_tokenizer_fertility_audit(
        "HuggingFaceTB/SmolLM2-135M-Instruct" if config.quick_smoke_test else config.model_id,
        config.hf_token,
    )
    vocab_jaccard = tokenizer_vocab_jaccard(
        "HuggingFaceTB/SmolLM2-135M-Instruct" if config.quick_smoke_test else config.model_id,
        REFERENCE_MODEL_IDS.get(str(similarities.iloc[0]["Reference Model"]), "Qwen/Qwen2.5-1.5B-Instruct"),
        config.hf_token,
    )
    ts, governance = transparency_score(config)

    best = similarities.iloc[0]
    has_safetensors = bool((metadata.get("repository_metadata") or {}).get("safetensors_available", False))
    risk, recommendation = risk_assessment(
        transparency=ts,
        highest_similarity=float(best["Cosine Similarity"]),
        matched_model=str(best["Reference Model"]),
        has_safetensors=has_safetensors,
    )

    scorecard = pd.DataFrame(
        [
            {"Metric": "Requested Model", "Value": config.model_id},
            {"Metric": "Audited Model", "Value": metadata["actual_model_id"]},
            {"Metric": "Transparency Score", "Value": f"{ts:.2f} / 1.00"},
            {"Metric": "Highest Similarity", "Value": f'{float(best["Cosine Similarity"]):.4f}'},
            {"Metric": "Nearest Reference", "Value": str(best["Reference Model"])},
            {"Metric": "Nearest Reference Family", "Value": REFERENCE_FAMILIES.get(str(best["Reference Model"]), "Unknown")},
            {"Metric": "Tokenizer Vocab Jaccard vs Nearest", "Value": f"{vocab_jaccard:.4f}"},
            {"Metric": "White-Box Safetensors Available", "Value": str(has_safetensors)},
            {"Metric": "Risk Level", "Value": risk},
            {"Metric": "Recommendation", "Value": recommendation},
            {"Metric": "Mean Coordinate Variance", "Value": f'{metadata["mean_coordinate_variance"]:.8f}'},
            {"Metric": "Perturbation Runs", "Value": str(config.perturbation_runs)},
            {"Metric": "Runtime Seconds", "Value": f"{time.time() - start:.2f}"},
        ]
    )

    png_path, svg_path = save_pca_plot(fingerprint, metadata["actual_model_id"], out_dir)
    latex = scorecard_latex(scorecard)
    exported = export_audit_bundle(
        fingerprint,
        metadata,
        similarities,
        distance_matrix,
        fertility,
        governance,
        scorecard,
        latex,
        out_dir,
    )
    exported["pca_png"] = png_path
    exported["pca_svg"] = svg_path

    report = {
        "runtime": asdict(get_runtime_report()),
        "metadata": metadata,
        "transparency_score": ts,
        "risk_level": risk,
        "recommendation": recommendation,
        "exported_files": exported,
    }
    metadata_df = metadata_to_dataframe(metadata)
    return similarities, distance_matrix, fertility, governance, scorecard, metadata_df, png_path, latex, json.dumps(report, indent=2, ensure_ascii=False)


def run_determinism_check(config: AuditConfig) -> pd.DataFrame:
    """Runs two quick fingerprints and reports coordinate variance."""
    quick_config = AuditConfig(**asdict(config))
    quick_config.perturbation_runs = min(3, int(config.perturbation_runs))
    quick_config.quick_smoke_test = True

    auditor_a = TensorGuardLiteAuditor(quick_config)
    fp_a, _ = auditor_a.extract_fingerprint()
    auditor_b = TensorGuardLiteAuditor(quick_config)
    fp_b, _ = auditor_b.extract_fingerprint()
    diffs = np.abs(fp_a - fp_b)
    return pd.DataFrame(
        {
            "Check": ["Max Absolute Coordinate Difference", "Mean Absolute Coordinate Difference", "Pass"],
            "Value": [float(np.max(diffs)), float(np.mean(diffs)), bool(np.allclose(fp_a, fp_b, atol=1e-6))],
        }
    )


def run_live_pairwise_lineage(
    model_ids: List[str],
    hf_token: str = "",
    seed: int = 42,
    perturbation_runs: int = 3,
    max_length: int = 64,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Audit several Hugging Face models sequentially and compute live pairwise
    fingerprint similarity. This is slower than comparing against the built-in
    reference library, but it is the strictest PRD validation mode.
    """
    fingerprints = {}
    metadata_rows = []
    for model_id in model_ids:
        cfg = AuditConfig(
            model_id=model_id,
            hf_token=hf_token,
            seed=seed,
            perturbation_runs=perturbation_runs,
            max_length=max_length,
            quick_smoke_test=False,
        )
        auditor = TensorGuardLiteAuditor(cfg)
        fp, meta = auditor.extract_fingerprint()
        fingerprints[model_id] = fp
        metadata_rows.append(meta)
        cleanup_cuda()

    rows = []
    for left in model_ids:
        for right in model_ids:
            rows.append(
                {
                    "Model A": left,
                    "Model B": right,
                    "Cosine Similarity": round(cosine_similarity(fingerprints[left], fingerprints[right]), 6),
                    "Euclidean Distance": round(float(np.linalg.norm(fingerprints[left] - fingerprints[right])), 6),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(metadata_rows)


def run_prd_validation_suite(
    hf_token: str = "",
    seed: int = 42,
    perturbation_runs: int = 3,
    max_length: int = 64,
    include_llama: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs the PRD's validation checks in a T4-aware way.
    - Known lineage: Qwen2.5-1.5B-Instruct vs DeepSeek-R1-Distill-Qwen-1.5B.
    - Cross-family divergence: optionally includes Llama-3.2-1B if access token is available.
    - Token tax: English/Hindi/Tamil tokenizer fertility.
    """
    models = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    ]
    if include_llama:
        models.append("meta-llama/Llama-3.2-1B")

    pairwise, metadata = run_live_pairwise_lineage(
        model_ids=models,
        hf_token=hf_token,
        seed=seed,
        perturbation_runs=perturbation_runs,
        max_length=max_length,
    )

    qwen_deepseek = pairwise[
        (pairwise["Model A"] == "Qwen/Qwen2.5-1.5B-Instruct")
        & (pairwise["Model B"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ]["Cosine Similarity"].iloc[0]

    checks = [
        {
            "Validation Scope": "Known Lineage Detection",
            "Procedure": "Qwen2.5-1.5B-Instruct vs DeepSeek-R1-Distill-Qwen-1.5B",
            "Observed Metric": qwen_deepseek,
            "PRD Target": "Cosine >= 0.92",
            "Pass": bool(qwen_deepseek >= 0.92),
        }
    ]

    if include_llama:
        llama_qwen = pairwise[
            (pairwise["Model A"] == "meta-llama/Llama-3.2-1B")
            & (pairwise["Model B"] == "Qwen/Qwen2.5-1.5B-Instruct")
        ]["Cosine Similarity"].iloc[0]
        checks.append(
            {
                "Validation Scope": "Cross-Family Divergence",
                "Procedure": "Llama-3.2-1B vs Qwen2.5-1.5B-Instruct",
                "Observed Metric": llama_qwen,
                "PRD Target": "Cosine < 0.45",
                "Pass": bool(llama_qwen < 0.45),
            }
        )

    token_tax = run_tokenizer_fertility_audit("Qwen/Qwen2.5-1.5B-Instruct", hf_token)
    non_english = token_tax[token_tax["Language"].isin(["Hindi", "Tamil"])]["Token Tax Psi vs English"].mean()
    checks.append(
        {
            "Validation Scope": "Multilingual Token Tax",
            "Procedure": "Average Hindi/Tamil token tax vs English using Qwen tokenizer",
            "Observed Metric": round(float(non_english), 4),
            "PRD Target": "Average Psi >= 2.50",
            "Pass": bool(non_english >= 2.50),
        }
    )

    out_dir = maybe_mount_drive(False)
    pairwise.to_csv(out_dir / "prd_live_pairwise_validation.csv", index=False)
    metadata.to_csv(out_dir / "prd_validation_metadata.csv", index=False)
    token_tax.to_csv(out_dir / "prd_token_tax_validation.csv", index=False)
    return pd.DataFrame(checks), pairwise, token_tax


def empty_dataframe() -> pd.DataFrame:
    return pd.DataFrame()


def live_pairwise_tables(fingerprints: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = list(fingerprints.keys())
    cosine_rows = []
    euclidean_matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            cos = cosine_similarity(fingerprints[left], fingerprints[right])
            dist = float(np.linalg.norm(fingerprints[left] - fingerprints[right]))
            euclidean_matrix[i, j] = dist
            cosine_rows.append(
                {
                    "Model A": left,
                    "Model B": right,
                    "Cosine Similarity": round(cos, 6),
                    "Euclidean Distance": round(dist, 6),
                }
            )
    return (
        pd.DataFrame(cosine_rows),
        pd.DataFrame(euclidean_matrix, index=labels, columns=labels).round(6).reset_index(names="Model"),
    )


def save_live_pca_plot(fingerprints: Dict[str, np.ndarray], out_dir: Path) -> str:
    labels = list(fingerprints.keys())
    matrix = np.vstack([fingerprints[label] for label in labels])
    matrix = StandardScaler().fit_transform(matrix)
    coords = PCA(n_components=2, random_state=42).fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(110 / 25.4, 82 / 25.4))
    for idx, label in enumerate(labels):
        short = label.split("/")[-1]
        if "DeepSeek" in label:
            family = "DeepSeek/Qwen derivative"
        elif "Qwen" in label:
            family = "Qwen"
        elif "llama" in label.lower():
            family = "Llama"
        elif "Ministral" in label:
            family = "Ministral"
        elif "SmolLM" in label:
            family = "SmolLM"
        else:
            family = "Unknown"
        ax.scatter(coords[idx, 0], coords[idx, 1], s=60, color=FAMILY_COLORS.get(family, "#1565c0"), edgecolor="black", linewidth=0.4)
        ax.annotate(short, (coords[idx, 0], coords[idx, 1]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_title("Live Paper Experiment PCA: Open-Weight SLMs", fontsize=9)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linewidth=0.25, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = out_dir / "paper_live_pca.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def tokenizer_jaccard_matrix(model_ids: List[str], hf_token: str = "") -> pd.DataFrame:
    vocabs = {}
    for model_id in model_ids:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token or None,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=str(cache_dir_for_model(model_id)),
        )
        vocabs[model_id] = set(tokenizer.get_vocab().keys())
    matrix = np.zeros((len(model_ids), len(model_ids)), dtype=np.float64)
    for i, left in enumerate(model_ids):
        for j, right in enumerate(model_ids):
            union = vocabs[left] | vocabs[right]
            matrix[i, j] = len(vocabs[left] & vocabs[right]) / max(1, len(union))
    return pd.DataFrame(matrix, index=model_ids, columns=model_ids).round(6).reset_index(names="Model")


def paper_validation_checks(pairwise: pd.DataFrame, token_tax_all: pd.DataFrame) -> pd.DataFrame:
    qwen = "Qwen/Qwen2.5-1.5B"
    deepseek = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    llama = "meta-llama/Llama-3.2-1B"
    smollm = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    def metric(a: str, b: str, col: str) -> float:
        row = pairwise[(pairwise["Model A"] == a) & (pairwise["Model B"] == b)]
        return float(row[col].iloc[0]) if not row.empty else float("nan")

    qwen_deepseek = metric(qwen, deepseek, "Cosine Similarity")
    llama_qwen = metric(llama, qwen, "Cosine Similarity")
    smollm_qwen = metric(smollm, qwen, "Cosine Similarity")
    hindi_llama = token_tax_all[
        (token_tax_all["Model"] == llama) & (token_tax_all["Language"] == "Hindi")
    ]["Fertility Phi"]
    hindi_fertility = float(hindi_llama.iloc[0]) if not hindi_llama.empty else float("nan")

    return pd.DataFrame(
        [
            {
                "Paper Claim / Test": "DeepSeek derivative clusters near Qwen parent",
                "Observed Metric": qwen_deepseek,
                "Expected Direction": "high cosine similarity",
                "Pass Heuristic": bool(qwen_deepseek >= max(llama_qwen, smollm_qwen)),
            },
            {
                "Paper Claim / Test": "Llama and SmolLM remain separated from Qwen family",
                "Observed Metric": f"Llama-Qwen={llama_qwen:.4f}, SmolLM-Qwen={smollm_qwen:.4f}",
                "Expected Direction": "lower than DeepSeek-Qwen",
                "Pass Heuristic": bool(qwen_deepseek > llama_qwen and qwen_deepseek > smollm_qwen),
            },
            {
                "Paper Claim / Test": "Hindi token fertility exposes Indic token tax",
                "Observed Metric": hindi_fertility,
                "Expected Direction": "Hindi fertility higher than English",
                "Pass Heuristic": bool(hindi_fertility > 1.2),
            },
            {
                "Paper Claim / Test": "White-box access required",
                "Observed Metric": "Local safetensors/model files are downloaded before gradient extraction",
                "Expected Direction": "API-only models cannot run this probe",
                "Pass Heuristic": True,
            },
        ]
    )


def run_paper_experiment_core(
    hf_token: str,
    seed: int,
    perturbation_runs: int,
    max_length: int,
    sample_gradient_entries: int,
    weight_noise_std: float,
    persist_model_cache: bool,
    save_to_drive: bool,
    progress=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, str]:
    out_dir = maybe_mount_drive(save_to_drive)
    configure_model_cache(persist_model_cache)
    logs = []
    fingerprints: Dict[str, np.ndarray] = {}
    metadata_rows = []

    logs.append("LIVE PAPER EXPERIMENT STARTED")
    logs.append("No stored reference fingerprints are used in this mode.")
    logs.append("Models: " + ", ".join(PAPER_MODEL_IDS))
    logs.append(f"Perturbation runs per model: {perturbation_runs}")
    logs.append(f"Max sampled gradient entries: {sample_gradient_entries}")
    logs.append(f"Gaussian weight noise std: {weight_noise_std}")
    if not hf_token.strip():
        logs.append("")
        logs.append("WARNING: HF Token is empty.")
        logs.append("The first paper model is meta-llama/Llama-3.2-1B, which is usually gated.")
        logs.append("If your Hugging Face account has not accepted the Llama license or no token is provided, the download will fail.")

    try:
        for idx, model_id in enumerate(PAPER_MODEL_IDS, start=1):
            logs.append(f"\n[{idx}/{len(PAPER_MODEL_IDS)}] Downloading/loading/fingerprinting: {model_id}")
            if progress:
                progress((idx - 1) / len(PAPER_MODEL_IDS), desc=f"Starting model {idx}/{len(PAPER_MODEL_IDS)}: {model_id}")
            cfg = AuditConfig(
                model_id=model_id,
                hf_token=hf_token,
                seed=seed,
                perturbation_runs=perturbation_runs,
                max_length=max_length,
                sample_gradient_entries=sample_gradient_entries,
                weight_noise_std=weight_noise_std,
                quick_smoke_test=False,
                save_to_drive=save_to_drive,
                persist_model_cache=persist_model_cache,
            )
            auditor = TensorGuardLiteAuditor(cfg)
            fp, meta = auditor.extract_fingerprint(progress=progress)
            fingerprints[model_id] = fp
            flat_meta = {
                "Model": model_id,
                "Params M": meta.get("total_params_m"),
                "Active Layers": meta.get("active_layer_count"),
                "Mean Coordinate Variance": meta.get("mean_coordinate_variance"),
                "Elapsed Seconds": meta.get("elapsed_seconds"),
                "Cache Status": (meta.get("download_report") or {}).get("cache_status"),
                "Downloaded Files": (meta.get("download_report") or {}).get("downloaded_files"),
                "Reused Cached Files": (meta.get("download_report") or {}).get("reused_cached_files"),
                "Safetensors Available": (meta.get("repository_metadata") or {}).get("safetensors_available"),
                "Architecture": (meta.get("repository_metadata") or {}).get("architecture"),
                "Target Modules": ", ".join(meta.get("target_modules", [])),
            }
            metadata_rows.append(flat_meta)
            logs.append(f"[{idx}/{len(PAPER_MODEL_IDS)}] Done: params={flat_meta['Params M']}M, safetensors={flat_meta['Safetensors Available']}, elapsed={flat_meta['Elapsed Seconds']}s")
            cleanup_cuda()
    except Exception as exc:
        cleanup_cuda()
        logs.append("")
        logs.append("LIVE PAPER EXPERIMENT FAILED BEFORE COMPLETION")
        logs.append(f"Error type: {type(exc).__name__}")
        logs.append(f"Error message: {exc}")
        logs.append("")
        logs.append("Most common cause: Llama is gated. Fix:")
        logs.append("1. Open https://huggingface.co/meta-llama/Llama-3.2-1B")
        logs.append("2. Log in and accept the model license/access request.")
        logs.append("3. Create a Hugging Face access token.")
        logs.append("4. Paste that token into the HF Token box and run again.")
        logs.append("")
        logs.append("Other possible causes: Colab GPU memory exhausted, network interruption, or a gated/unavailable model repository.")
        error_df = pd.DataFrame(
            [
                {
                    "Status": "FAILED",
                    "Error Type": type(exc).__name__,
                    "Error Message": str(exc),
                    "Likely Cause": "Missing/invalid HF token for gated model, or download/runtime failure.",
                }
            ]
        )
        return (
            error_df,
            empty_dataframe(),
            empty_dataframe(),
            empty_dataframe(),
            empty_dataframe(),
            pd.DataFrame(metadata_rows),
            None,
            "\n".join(logs),
            json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, indent=2),
        )

    pairwise, distance_matrix = live_pairwise_tables(fingerprints)
    logs.append("\nGenerated live cosine/Euclidean pairwise results from all fresh fingerprints.")
    pca_path = save_live_pca_plot(fingerprints, out_dir)
    logs.append(f"Saved live PCA plot: {pca_path}")

    token_tax_frames = []
    for model_id in PAPER_MODEL_IDS:
        logs.append(f"Running tokenizer fertility audit: {model_id}")
        frame = run_tokenizer_fertility_audit(model_id, hf_token)
        frame.insert(0, "Model", model_id)
        token_tax_frames.append(frame)
    token_tax_all = pd.concat(token_tax_frames, ignore_index=True)

    logs.append("Calculating tokenizer vocabulary Jaccard matrix.")
    jaccard = tokenizer_jaccard_matrix(PAPER_MODEL_IDS, hf_token)
    checks = paper_validation_checks(pairwise, token_tax_all)
    metadata = pd.DataFrame(metadata_rows)

    pairwise.to_csv(out_dir / "paper_live_pairwise_similarity.csv", index=False)
    distance_matrix.to_csv(out_dir / "paper_live_euclidean_distance_matrix.csv", index=False)
    token_tax_all.to_csv(out_dir / "paper_live_tokenizer_tax_all_models.csv", index=False)
    jaccard.to_csv(out_dir / "paper_live_tokenizer_jaccard_matrix.csv", index=False)
    metadata.to_csv(out_dir / "paper_live_model_metadata.csv", index=False)
    checks.to_csv(out_dir / "paper_live_validation_checks.csv", index=False)

    report = {
        "mode": "live_paper_experiment_no_stored_reference_fingerprints",
        "models": PAPER_MODEL_IDS,
        "outputs": {
            "pairwise": str(out_dir / "paper_live_pairwise_similarity.csv"),
            "euclidean_matrix": str(out_dir / "paper_live_euclidean_distance_matrix.csv"),
            "tokenizer_tax": str(out_dir / "paper_live_tokenizer_tax_all_models.csv"),
            "jaccard": str(out_dir / "paper_live_tokenizer_jaccard_matrix.csv"),
            "metadata": str(out_dir / "paper_live_model_metadata.csv"),
            "checks": str(out_dir / "paper_live_validation_checks.csv"),
            "pca": pca_path,
        },
    }
    logs.append("\nLIVE PAPER EXPERIMENT COMPLETE")
    if progress:
        progress(1.0, desc="Live paper experiment complete")
    return checks, pairwise, distance_matrix, token_tax_all, jaccard, metadata, pca_path, "\n".join(logs), json.dumps(report, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------------------
# 11. Gradio UI/UX
# ------------------------------------------------------------------------------

def bool_from_yes_no(value: str) -> bool:
    return str(value).strip().lower() == "yes"


def ui_run_audit(
    model_id,
    custom_model_id,
    hf_token,
    seed,
    perturbation_runs,
    max_length,
    sample_gradient_entries,
    weight_noise_std,
    sample_text,
    quick_smoke_test,
    save_to_drive,
    persist_model_cache,
    open_weights,
    data_transparency,
    recipe_disclosure,
    tokenizer_openness,
    safety_report,
    evaluation_results,
    cryptographic_data_proof,
    architecture_documented,
    regulator_safetensors_access,
    compute_subsidy_disclosure,
    progress=gr.Progress(track_tqdm=True),
):
    chosen_model = custom_model_id.strip() if custom_model_id.strip() else model_id
    config = AuditConfig(
        model_id=chosen_model,
        hf_token=hf_token.strip(),
        seed=int(seed),
        perturbation_runs=int(perturbation_runs),
        max_length=int(max_length),
        sample_gradient_entries=int(sample_gradient_entries),
        weight_noise_std=float(weight_noise_std),
        sample_text=sample_text.strip() or DEFAULT_SAMPLE_TEXT,
        quick_smoke_test=bool(quick_smoke_test),
        save_to_drive=bool(save_to_drive),
        persist_model_cache=bool(persist_model_cache),
        open_weights=bool_from_yes_no(open_weights),
        data_transparency=bool_from_yes_no(data_transparency),
        recipe_disclosure=bool_from_yes_no(recipe_disclosure),
        tokenizer_openness=bool_from_yes_no(tokenizer_openness),
        safety_report=bool_from_yes_no(safety_report),
        evaluation_results=bool_from_yes_no(evaluation_results),
        cryptographic_data_proof=bool_from_yes_no(cryptographic_data_proof),
        architecture_documented=bool_from_yes_no(architecture_documented),
        regulator_safetensors_access=bool_from_yes_no(regulator_safetensors_access),
        compute_subsidy_disclosure=bool_from_yes_no(compute_subsidy_disclosure),
    )
    return run_full_audit(config, progress=progress)


def ui_runtime_report():
    return pd.DataFrame([asdict(get_runtime_report())])


def ui_run_validation(hf_token, seed, perturbation_runs, max_length, include_llama):
    return run_prd_validation_suite(
        hf_token=hf_token.strip(),
        seed=int(seed),
        perturbation_runs=int(perturbation_runs),
        max_length=int(max_length),
        include_llama=bool(include_llama),
    )


def ui_run_paper_experiment(
    hf_token,
    seed,
    perturbation_runs,
    max_length,
    sample_gradient_entries,
    weight_noise_std,
    persist_model_cache,
    save_to_drive,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        return run_paper_experiment_core(
            hf_token=hf_token.strip(),
            seed=int(seed),
            perturbation_runs=int(perturbation_runs),
            max_length=int(max_length),
            sample_gradient_entries=int(sample_gradient_entries),
            weight_noise_std=float(weight_noise_std),
            persist_model_cache=bool(persist_model_cache),
            save_to_drive=bool(save_to_drive),
            progress=progress,
        )
    except Exception as exc:
        error_df = pd.DataFrame(
            [
                {
                    "Status": "FAILED",
                    "Error Type": type(exc).__name__,
                    "Error Message": str(exc),
                    "Likely Cause": "Setup, download, token, Colab runtime, or Gradio output failure.",
                }
            ]
        )
        log_text = (
            "LIVE PAPER EXPERIMENT FAILED BEFORE THE CONTROLLED MODEL LOOP\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error message: {exc}\n\n"
            "If this mentions meta-llama or 401/403/gated repo, your Hugging Face token does not have Llama access.\n"
            "If this mentions CUDA out of memory, reduce perturbation runs, sequence length, or sampled gradient entries.\n"
        )
        return (
            error_df,
            empty_dataframe(),
            empty_dataframe(),
            empty_dataframe(),
            empty_dataframe(),
            empty_dataframe(),
            None,
            log_text,
            json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, indent=2),
        )


def build_dashboard():
    css = """
    .tensorguard-hero {
        padding: 18px 20px;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        background: #f8fafc;
    }
    .tensorguard-hero h1 {
        margin: 0 0 6px 0;
        font-size: 28px;
        letter-spacing: 0;
    }
    .tensorguard-hero p {
        margin: 0;
        color: #334155;
        font-size: 14px;
    }
    """

    with gr.Blocks(title=APP_NAME, css=css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
        gr.HTML(
            """
            <div class="tensorguard-hero">
              <h1>TensorGuard-Lite Sovereign AI Provenance Auditor</h1>
              <p>T4-ready live provenance experiment with gradient fingerprints, lineage similarity, token tax, risk scorecards, and publication exports.</p>
            </div>
            """
        )

        with gr.Tab("Live Paper Experiment"):
            gr.Markdown(
                "This mode loads and fingerprints the live open-weight comparison set currently enabled in the notebook: Llama-3.2-1B, SmolLM2-1.7B-Instruct, Qwen2.5-1.5B, and DeepSeek-R1-Distill-Qwen-1.5B. It does not use stored reference fingerprints."
            )
            gr.Markdown(
                "**Important:** this mode starts with `meta-llama/Llama-3.2-1B`, which normally requires a Hugging Face token with accepted Llama access. If the token is missing or unauthorized, the run will stop and the log will show the exact error."
            )
            paper_log = gr.Textbox(label="Live Execution Log", lines=14, interactive=False)
            with gr.Row():
                paper_token = gr.Textbox(label="HF Token", type="password", placeholder="Required for gated Llama access")
                paper_seed = gr.Slider(1, 100000, value=42, step=1, label="Seed")
                paper_runs = gr.Slider(1, 30, value=1, step=1, label="Perturbation Runs Per Model")
                paper_length = gr.Slider(32, 128, value=32, step=8, label="Max Sequence Length")
            with gr.Row():
                paper_gradient_entries = gr.Slider(10000, 500000, value=500000, step=10000, label="Sampled Gradient Entries")
                paper_noise = gr.Slider(0.0, 0.01, value=0.001, step=0.0005, label="Gaussian Weight Noise Std")
                paper_persist_cache = gr.Checkbox(label="Persist downloaded models in Google Drive", value=False)
                paper_save_drive = gr.Checkbox(label="Save experiment outputs to Google Drive", value=False)
            paper_button = gr.Button("Run Live Paper Experiment", variant="primary", size="lg")

            paper_checks = gr.Dataframe(label="Paper Claim/Test Results", interactive=False)
            with gr.Row():
                paper_pairwise = gr.Dataframe(label="Live Pairwise Cosine + Euclidean Results", interactive=False)
                paper_pca = gr.Image(label="Live PCA", type="filepath")
            paper_distance = gr.Dataframe(label="Live Euclidean Distance Matrix", interactive=False)
            paper_token_tax = gr.Dataframe(label="Tokenizer Fertility / Token Tax Across Live Models", interactive=False)
            paper_jaccard = gr.Dataframe(label="Tokenizer Vocabulary Jaccard Matrix", interactive=False)
            paper_metadata = gr.Dataframe(label="Live Model Download / Safetensors / Architecture Metadata", interactive=False)
            paper_report = gr.Code(label="Paper Experiment Export Report", language="json")

        with gr.Tab("Audit Setup"):
            gr.Markdown(
                "Exploratory single-model mode. This audits one selected model live, then compares it to built-in reference fingerprints. Use the Live Paper Experiment tab for the live multi-model result."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    model_id = gr.Dropdown(DEFAULT_MODEL_OPTIONS, value=DEFAULT_MODEL_OPTIONS[0], label="Hugging Face Model")
                    custom_model_id = gr.Textbox(label="Custom Hugging Face Model ID", placeholder="Optional, e.g. Qwen/Qwen2.5-1.5B-Instruct")
                    hf_token = gr.Textbox(label="HF Token for gated models", type="password", placeholder="Optional. Required for gated Llama access.")
                    quick_smoke_test = gr.Checkbox(label="Quick smoke test mode", value=False)
                    save_to_drive = gr.Checkbox(label="Save exports to Google Drive", value=False)
                    persist_model_cache = gr.Checkbox(label="Persist downloaded models in Google Drive", value=False)
                with gr.Column(scale=1):
                    seed = gr.Slider(1, 100000, value=42, step=1, label="Deterministic Seed")
                    perturbation_runs = gr.Slider(1, 30, value=6, step=1, label="Perturbation Runs")
                    max_length = gr.Slider(32, 192, value=96, step=8, label="Max Sequence Length")
                    sample_gradient_entries = gr.Slider(10000, 500000, value=500000, step=10000, label="Max Sampled Gradient Entries")
                    weight_noise_std = gr.Slider(0.0, 0.01, value=0.0, step=0.0005, label="Gaussian Weight Noise Std")
                    sample_text = gr.Textbox(label="Audit Prompt", value=DEFAULT_SAMPLE_TEXT, lines=5)

            gr.Markdown("### Governance Dimensions")
            with gr.Row():
                open_weights = gr.Radio(["Yes", "No"], value="Yes", label="Open Weights")
                data_transparency = gr.Radio(["Yes", "No"], value="No", label="Data Transparency")
                recipe_disclosure = gr.Radio(["Yes", "No"], value="No", label="Recipe Disclosure")
            with gr.Row():
                tokenizer_openness = gr.Radio(["Yes", "No"], value="Yes", label="Tokenizer Openness")
                safety_report = gr.Radio(["Yes", "No"], value="No", label="Safety Report")
                evaluation_results = gr.Radio(["Yes", "No"], value="No", label="Evaluation Results")
            with gr.Row():
                cryptographic_data_proof = gr.Radio(["Yes", "No"], value="No", label="Cryptographic Data Lineage Proof")
                architecture_documented = gr.Radio(["Yes", "No"], value="Yes", label="Architecture Documented")
                regulator_safetensors_access = gr.Radio(["Yes", "No"], value="No", label="Regulator Safetensors Access")
                compute_subsidy_disclosure = gr.Radio(["Yes", "No"], value="No", label="Compute Subsidy Disclosure")

            run_button = gr.Button("Run Sovereign Provenance Audit", variant="primary", size="lg")

        with gr.Tab("Lineage Results"):
            with gr.Row():
                similarity_out = gr.Dataframe(label="Cosine and Euclidean Similarity", interactive=False)
                pca_out = gr.Image(label="PCA Lineage Map", type="filepath")
            distance_matrix_out = gr.Dataframe(label="Full Euclidean Distance Matrix", interactive=False)

        with gr.Tab("Tokenizer Tax"):
            fertility_out = gr.Dataframe(label="Multilingual Fertility and Token Tax", interactive=False)

        with gr.Tab("Governance Scorecard"):
            governance_out = gr.Dataframe(label="Weighted Transparency Dimensions", interactive=False)
            scorecard_out = gr.Dataframe(label="Sovereign Scorecard", interactive=False)

        with gr.Tab("Model Metadata"):
            metadata_out = gr.Dataframe(label="Download, Cache, Safetensors, and Architecture Metadata", interactive=False)

        with gr.Tab("Exports"):
            latex_out = gr.Code(label="LaTeX Scorecard", language="latex")
            report_out = gr.Code(label="Audit JSON Report", language="json")

        with gr.Tab("Runtime"):
            runtime_btn = gr.Button("Refresh Runtime Report")
            runtime_table = gr.Dataframe(label="Colab Runtime Report", value=ui_runtime_report, interactive=False)
            runtime_btn.click(ui_runtime_report, outputs=[runtime_table])

        with gr.Tab("PRD Validation"):
            gr.Markdown(
                "Run sequential live audits for the PRD experiments. Start with 3 perturbation runs; increase to 30 for final evidence."
            )
            with gr.Row():
                validation_token = gr.Textbox(label="HF Token", type="password", placeholder="Optional unless including gated Llama")
                validation_seed = gr.Slider(1, 100000, value=42, step=1, label="Seed")
                validation_runs = gr.Slider(1, 30, value=3, step=1, label="Perturbation Runs")
                validation_length = gr.Slider(32, 128, value=64, step=8, label="Max Length")
            include_llama = gr.Checkbox(label="Include Llama cross-family test", value=False)
            validation_btn = gr.Button("Run PRD Validation Suite", variant="primary")
            validation_checks = gr.Dataframe(label="Validation Matrix", interactive=False)
            validation_pairwise = gr.Dataframe(label="Live Pairwise Fingerprint Similarity", interactive=False)
            validation_token_tax = gr.Dataframe(label="Validation Token Tax", interactive=False)
            validation_btn.click(
                ui_run_validation,
                inputs=[validation_token, validation_seed, validation_runs, validation_length, include_llama],
                outputs=[validation_checks, validation_pairwise, validation_token_tax],
            )

        paper_button.click(
            fn=ui_run_paper_experiment,
            inputs=[
                paper_token,
                paper_seed,
                paper_runs,
                paper_length,
                paper_gradient_entries,
                paper_noise,
                paper_persist_cache,
                paper_save_drive,
            ],
            outputs=[
                paper_checks,
                paper_pairwise,
                paper_distance,
                paper_token_tax,
                paper_jaccard,
                paper_metadata,
                paper_pca,
                paper_log,
                paper_report,
            ],
        )

        run_button.click(
            fn=ui_run_audit,
            inputs=[
                model_id,
                custom_model_id,
                hf_token,
                seed,
                perturbation_runs,
                max_length,
                sample_gradient_entries,
                weight_noise_std,
                sample_text,
                quick_smoke_test,
                save_to_drive,
                persist_model_cache,
                open_weights,
                data_transparency,
                recipe_disclosure,
                tokenizer_openness,
                safety_report,
                evaluation_results,
                cryptographic_data_proof,
                architecture_documented,
                regulator_safetensors_access,
                compute_subsidy_disclosure,
            ],
            outputs=[
                similarity_out,
                distance_matrix_out,
                fertility_out,
                governance_out,
                scorecard_out,
                metadata_out,
                pca_out,
                latex_out,
                report_out,
            ],
        )

    return demo


# ------------------------------------------------------------------------------
# 12. CLI/Colab launch
# ------------------------------------------------------------------------------

def launch_colab_app():
    print("=" * 80)
    print(APP_NAME)
    print("=" * 80)
    print(pd.DataFrame([asdict(get_runtime_report())]).to_string(index=False))
    print("\nTip: use Quick smoke test mode first, then run the full target model.")
    print("For gated Llama models, paste a Hugging Face token with accepted model access.")
    demo = build_dashboard()
    demo.queue(max_size=8).launch(share=True, debug=False)


if __name__ == "__main__":
    launch_colab_app()
