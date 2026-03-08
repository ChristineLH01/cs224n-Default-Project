# modal run modal_train_sonnet.py --epochs 25 --lr 2e-5 --model-size gpt2 --fine-tune-mode lora --lora-rank 8 --lora-alpha 8 --use-flash-attention
import modal
import os

# Local project directory (where this script lives)
LOCAL_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the Modal app
app = modal.App("cs224n-sonnet")

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
    gpu="H100",  # Options: "T4", "L4", "A10G", "A100", "H100"
    timeout=3600,  # 1 hour timeout
)
def train_sonnet(
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
    adalora_alpha: float = 8.0,
    adalora_init_r: int = 12,
    adalora_target_r: int = 4,
    adalora_tinit_ratio: float = 0.2,
    adalora_deltat: int = 10,
):
    import subprocess
    import sys

    os.chdir("/root/project")

    # Build the command
    cmd = [
        "python", "-u", "sonnet_generation.py",
        "--fine_tune_mode", str(fine_tune_mode),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--model_size", str(model_size),
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
        "--adalora_alpha", str(adalora_alpha),  
        "--adalora_init_r", str(adalora_init_r), 
        "--adalora_target_r", str(adalora_target_r),  
        "--adalora_tinit_ratio", str(adalora_tinit_ratio),
        "--adalora_deltaT", str(adalora_deltat),    
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
    existing_txt = set()
    if os.path.exists(predictions_dir):
        existing_txt = set(f for f in os.listdir(predictions_dir) if f.endswith(".txt"))

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

    # Collect only NEW prediction .txt files
    prediction_files = {}
    if os.path.exists(predictions_dir):
        for f in os.listdir(predictions_dir):
            if f.endswith(".txt") and f not in existing_txt:
                with open(f"{predictions_dir}/{f}", "rb") as fh:
                    prediction_files[f] = fh.read()
                print(f"Collected prediction: {f}")

    print("Training complete!")
    return model_files, prediction_files


@app.local_entrypoint()
def main(
    fine_tune_mode: str = "full-model",
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 8,
    model_size: str = "gpt2-medium",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
    use_flash_attention: bool = False,
    use_rslora: bool = False,
    adalora_alpha: float = 8.0,
    adalora_init_r: int = 12,
    adalora_target_r: int = 4,
    adalora_tinit_ratio: float = 0.2,
    adalora_deltat: int = 10,
):
    print(f"Starting sonnet generation training on Modal with GPU...")
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
    print(f"  adalora_deltat: {adalora_deltat}")
    print(f"  adalora_tinit_ratio: {adalora_tinit_ratio}")

    model_files, prediction_files = train_sonnet.remote(
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
        adalora_deltat=adalora_deltat,
    )

    # Save .pt model files to the project folder
    for name, data in model_files.items():
        out_path = os.path.join(LOCAL_PROJECT_DIR, name)
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved model: {out_path}")

    # Save prediction TXT files to the predictions/ subfolder
    predictions_dir = os.path.join(LOCAL_PROJECT_DIR, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    for name, data in prediction_files.items():
        out_path = os.path.join(predictions_dir, name)
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved prediction: {out_path}")