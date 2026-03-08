import modal
import os

# Local project directory (where this script lives)
LOCAL_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the Modal app
app = modal.App("cs224n-paraphrase")

# Create a Modal image with all dependencies and local project files
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "tqdm==4.58.0",
        "requests==2.25.1",
        "importlib-metadata==3.7.0",
        "filelock==3.0.12",
        "scikit-learn",
        "tokenizers==0.20",
        "einops==0.8.0",
        "transformers==4.46.3",
        "numpy<2",
        "sacrebleu",
    )
    .add_local_dir(
        LOCAL_PROJECT_DIR,
        remote_path="/root/project",
        ignore=[".git", "__pycache__", "*.pt", "node_modules"],
    )
)


@app.function(
    image=image,
    gpu="T4",  # Options: "T4", "L4", "A10G", "A100", "H100"
    timeout=3600,  # 1 hour timeout
)
def train_paraphrase(
    fine_tune_mode: str = "full-model",
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 8,
    model_size: str = "gpt2-medium",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
    use_flash_attention: bool = False,
    use_gpu: bool = True,
    use_rslora: bool = False,
    adalora_alpha: float = 16.0,
    adalora_init_r: int = 16,
    adalora_target_r: int = 8,
    adalora_tinit_ratio: float = 0.05,
    adalora_delta_t: int = 50,
):
    import subprocess
    import sys

    os.chdir("/root/project")

    # Build the command
    cmd = [
        "python", "-u", "paraphase_detection.py",
        "--fine_tune_mode", fine_tune_mode,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--model_size", model_size,
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
        "--adalora_alpha", str(adalora_alpha),
        "--adalora_init_r", str(adalora_init_r),
        "--adalora_target_r", str(adalora_target_r),
        "--adalora_tinit_ratio", str(adalora_tinit_ratio),
        "--adalora_delta_t", str(adalora_delta_t),
    ]
    if use_gpu:
        cmd.append("--use_gpu")
    if use_flash_attention:
        cmd.append("--use_flash_attention")
    if use_rslora:
        cmd.append("--use_rslora")

    print(f"Running: {' '.join(cmd)}")
    print(f"GPU available: {__import__('torch').cuda.is_available()}")
    if __import__('torch').cuda.is_available():
        print(f"GPU: {__import__('torch').cuda.get_device_name(0)}")

    # Record existing files before training so we only collect new ones
    existing_pt = set(f for f in os.listdir("/root/project") if f.endswith(".pt"))
    predictions_dir = "/root/project/predictions"
    existing_csv = set()
    if os.path.exists(predictions_dir):
        existing_csv = set(f for f in os.listdir(predictions_dir) if f.endswith(".csv"))

    # Run training (stream output in real-time via print so Modal captures it)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True
    )
    output_lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    process.wait()

    if process.returncode != 0:
        # Show last 50 lines of output for debugging
        tail = "".join(output_lines[-50:])
        raise RuntimeError(
            f"Training failed with return code {process.returncode}\n"
            f"--- Last output ---\n{tail}"
        )

    # Collect only NEW model .pt files
    model_files = {}
    for f in os.listdir("/root/project"):
        if f.endswith(".pt") and f not in existing_pt:
            with open(f"/root/project/{f}", "rb") as fh:
                model_files[f] = fh.read()
            print(f"Collected model: {f}")

    # Collect only NEW prediction .csv files
    prediction_files = {}
    if os.path.exists(predictions_dir):
        for f in os.listdir(predictions_dir):
            if f.endswith(".csv") and f not in existing_csv:
                with open(f"{predictions_dir}/{f}", "rb") as fh:
                    prediction_files[f] = fh.read()
                print(f"Collected prediction: {f}")

    print("Training complete!")
    return model_files, prediction_files


@app.local_entrypoint()
def main(
    fine_tune_mode: str = "last-linear-layer",
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 8,
    model_size: str = "gpt2-medium",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
    use_flash_attention: bool = False,
    use_rslora: bool = False,
    adalora_alpha: float = 16.0,
    adalora_init_r: int = 16,
    adalora_target_r: int = 8,
    adalora_tinit_ratio: float = 0.05,
    adalora_delta_t: int = 50,
):
    print(f"Starting paraphrase detection training on Modal with GPU...")
    print(f"  fine_tune_mode: {fine_tune_mode}")
    print(f"  epochs: {epochs}")
    print(f"  lr: {lr}")
    print(f"  batch_size: {batch_size}")
    print(f"  model_size: {model_size}")
    print(f"  lora_rank: {lora_rank}")
    print(f"  lora_alpha: {lora_alpha}")
    print(f"  use_flash_attention: {use_flash_attention}")
    print(f"  use_rslora: {use_rslora}")
    print(f"  adalora_alpha: {adalora_alpha}")
    print(f"  adalora_init_r: {adalora_init_r}")
    print(f"  adalora_target_r: {adalora_target_r}")
    print(f"  adalora_delta_t: {adalora_delta_t}")
    print(f"  adalora_tinit_ratio: {adalora_tinit_ratio}")

    model_files, prediction_files = train_paraphrase.remote(
        fine_tune_mode=fine_tune_mode,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        model_size=model_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_flash_attention=use_flash_attention,
        use_gpu=True,
        use_rslora=use_rslora,
        adalora_alpha=adalora_alpha,
        adalora_init_r=adalora_init_r,
        adalora_target_r=adalora_target_r,
        adalora_tinit_ratio=adalora_tinit_ratio,
        adalora_delta_t=adalora_delta_t,
    )

    # Save .pt model files to the project folder
    for name, data in model_files.items():
        out_path = os.path.join(LOCAL_PROJECT_DIR, name)
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved model: {out_path}")

    # Save prediction CSVs to the predictions/ subfolder
    predictions_dir = os.path.join(LOCAL_PROJECT_DIR, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    for name, data in prediction_files.items():
        out_path = os.path.join(predictions_dir, name)
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved prediction: {out_path}")
