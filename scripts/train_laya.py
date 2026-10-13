#!/usr/bin/env python3
"""
scripts/train_laya.py
Orchestrates fine-tuning of Laya on local GPU via `laya-train`.
Configures RLCD loss, option shuffling, micro-batch gradient accumulation,
and post-hoc temperature calibration.

Inputs:  data/train.jsonl (and optional data/test.jsonl for evaluation)
Outputs: checkpoints/laya_finetuned/ (model weights, heads, temperatures, config)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Prevent CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def check_docker_available() -> bool:
    """Check if docker compose has a running laya-serve container."""
    try:
        res = subprocess.run(
            ["docker", "compose", "ps", "--services", "--filter", "status=running"],
            capture_output=True,
            text=True,
            check=True
        )
        return "laya-serve" in res.stdout
    except Exception:
        return False


def run_training(
    train_data: str = "data/train.jsonl",
    eval_data: str = "data/test.jsonl",
    output_dir: str = "checkpoints/laya_finetuned",
    base_model: str = "convaiinnovations/laya",
    epochs: int = 4,
    micro_batch: int = 4,
    grad_accum: int = 16,
    encoder_lr: float = 2.5e-5,
    head_lr: float = 1e-4,
    loss: str = "rlcd",
    shuffle_options: bool = True,
    freeze_encoder: bool = False,
    device: str = "auto",
    runner: str = "auto",
    dry_run: bool = False,
):
    train_file = Path(train_data)
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}. Run label_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine execution mode: local or docker
    import shutil
    has_local = shutil.which("laya-train") is not None
    docker_running = check_docker_available()

    use_docker = False
    if runner == "docker":
        use_docker = True
    elif runner == "local":
        use_docker = False
    else:  # auto
        if not has_local and docker_running:
            use_docker = True
        elif has_local:
            use_docker = False
        elif docker_running:
            use_docker = True

    if use_docker:
        # In Docker container, /data and /checkpoints are mounted
        c_train = f"/data/{train_file.name}"
        c_out = f"/checkpoints/{out_path.name}"
        cmd = [
            "docker", "compose", "run", "--rm", "laya-serve",
            "laya-train",
            "--data", c_train,
            "--base", base_model,
            "--out", c_out,
            "--epochs", str(epochs),
            "--micro-batch", str(micro_batch),
            "--grad-accum", str(grad_accum),
            "--encoder-lr", str(encoder_lr),
            "--head-lr", str(head_lr),
            "--loss", loss,
            "--device", "cuda" if device in ("auto", "cuda") else device,
        ]
        if eval_data and Path(eval_data).exists():
            cmd.extend(["--eval", f"/data/{Path(eval_data).name}"])
        if shuffle_options:
            cmd.append("--shuffle-options")
        if freeze_encoder:
            cmd.append("--freeze-encoder")
        if dry_run:
            cmd.append("--dry-run")
        runner_label = "Docker Run Container (laya-serve [CUDA])"
    else:
        cmd = [
            "laya-train",
            "--data", str(train_file.resolve()),
            "--base", base_model,
            "--out", str(out_path.resolve()),
            "--epochs", str(epochs),
            "--micro-batch", str(micro_batch),
            "--grad-accum", str(grad_accum),
            "--encoder-lr", str(encoder_lr),
            "--head-lr", str(head_lr),
            "--loss", loss,
            "--device", device,
        ]
        if eval_data and Path(eval_data).exists():
            cmd.extend(["--eval", str(Path(eval_data).resolve())])
        if shuffle_options:
            cmd.append("--shuffle-options")
        if freeze_encoder:
            cmd.append("--freeze-encoder")
        if dry_run:
            cmd.append("--dry-run")
        runner_label = "Host Python Environment"

    print("\n" + "=" * 60)
    print("Launching Laya Local Fine-Tuning")
    print("=" * 60)
    print(f"  • Execution Mode:  {runner_label}")
    print(f"  • Training Data:   {train_file}")
    print(f"  • Base Checkpoint: {base_model}")
    print(f"  • Output Directory:{out_path}")
    print(f"  • Objective Loss:  {loss.upper()}")
    print(f"  • Encoder Weights: {'Frozen (Head fine-tuning, fits <=8GB GPU)' if freeze_encoder else 'Trainable (Full fine-tuning)'}")
    print(f"  • Epochs:          {epochs}")
    print(f"  • Effective Batch: {micro_batch * grad_accum} ({micro_batch} micro x {grad_accum} accum)")
    print(f"  • Learning Rates:  Encoder={encoder_lr} | Head={head_lr}")
    print(f"  • Option Shuffle:  {'Enabled' if shuffle_options else 'Disabled'}")
    print(f"  • Command:         {' '.join(cmd)}")
    print("=" * 60 + "\n")

    was_serving = docker_running and not dry_run
    if was_serving:
        print("Pausing background laya-serve container to free GPU VRAM for fine-tuning...")
        subprocess.run(["docker", "compose", "stop", "laya-serve"], check=False)

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training completed successfully in {elapsed:.1f}s ({elapsed/60.0:.2f} mins)!")
        print(f"Saved checkpoint artifacts to {out_path.resolve()}")
        print("=" * 60 + "\n")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with return code {e.returncode}.", file=sys.stderr)
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\nError: `laya-train` executable not found on PATH and Docker is unavailable.", file=sys.stderr)
        print("Either run Docker (`docker compose up -d laya-serve`) or install laya locally: `pip install -e /home/vinayb/apps/laya`", file=sys.stderr)
        sys.exit(1)
    finally:
        if was_serving:
            print("Restoring background laya-serve container...")
            subprocess.run(["docker", "compose", "up", "-d", "laya-serve"], check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/fine-tune local Laya model on social decision tasks")
    parser.add_argument("--train-data", type=str, default="data/train.jsonl", help="Training JSONL dataset")
    parser.add_argument("--eval-data", type=str, default="data/test.jsonl", help="Evaluation JSONL dataset")
    parser.add_argument("--output-dir", type=str, default="checkpoints/laya_finetuned", help="Output directory")
    parser.add_argument("--base-model", type=str, default="convaiinnovations/laya", help="Base model identifier")
    parser.add_argument("--epochs", type=int, default=4, help="Epoch count (default: 4)")
    parser.add_argument("--micro-batch", type=int, default=4, help="Micro-batch size (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps (default: 16)")
    parser.add_argument("--encoder-lr", type=float, default=2.5e-5, help="Encoder learning rate")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="Head learning rate")
    parser.add_argument("--loss", choices=["rlcd", "soft-ce"], default="rlcd", help="Training loss")
    parser.add_argument("--device", default="auto", help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--no-shuffle-options", action="store_true", help="Disable option order shuffling")
    parser.add_argument("--freeze-encoder", action="store_true", default=True, help="Freeze encoder backbone (fits <=8GB GPU, default: True)")
    parser.add_argument("--unfreeze-encoder", action="store_false", dest="freeze_encoder", help="Train full encoder (requires >=16GB VRAM)")
    parser.add_argument("--runner", choices=["auto", "docker", "local"], default="auto", help="Execution environment (auto: prefer local if present, else docker container)")
    parser.add_argument("--dry-run", action="store_true", help="Inspect dataset and print stats without training")
    args = parser.parse_args()

    run_training(
        train_data=args.train_data,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        loss=args.loss,
        shuffle_options=not args.no_shuffle_options,
        freeze_encoder=args.freeze_encoder,
        device=args.device,
        runner=args.runner,
        dry_run=args.dry_run,
    )
