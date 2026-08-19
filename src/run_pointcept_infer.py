import argparse
import os
import subprocess
from pathlib import Path


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Call Pointcept tools/test.py for semantic segmentation")
    parser.add_argument(
        "--pointcept-dir",
        type=Path,
        default=repo_root / "third_party" / "pointcept",
        help="Pointcept repo root (must contain tools/test.py)",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        required=True,
        help="Pointcept config, e.g. third_party/pointcept/configs/scannet/semseg-pt-v3m1-0-base.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Pointcept checkpoint (.pth)",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Dataset root produced by prepare_pointcept_input.py (contains <split>/<scene>/coord.npy)",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Split folder inside --input-root",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=repo_root / "outputs" / "pointcept",
        help="Prediction output directory",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataloader workers (config default of 24 is tuned for training)",
    )
    parser.add_argument(
        "--enable-flash",
        action="store_true",
        help="Keep flash attention on (requires flash_attn to be installed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command only",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pointcept_root = args.pointcept_dir.resolve()
    tools_test = pointcept_root / "tools" / "test.py"
    if not tools_test.exists():
        raise FileNotFoundError(
            f"Cannot find {tools_test}. Clone Pointcept first:\n"
            "  git clone https://github.com/Pointcept/Pointcept.git third_party/pointcept"
        )
    if not args.config_file.exists():
        raise FileNotFoundError(f"Config not found: {args.config_file}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    split_dir = args.input_root / args.split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Split folder not found: {split_dir}\n"
            "  Run src/prepare_pointcept_input.py first"
        )

    args.save_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(pointcept_root)

    data_root = args.input_root.resolve()
    options = [
        f"save_path={args.save_path.resolve()}",
        f"weight={args.checkpoint.resolve()}",
        f"num_worker={args.num_workers}",
        "enable_wandb=false",
        f"data_root={data_root}",
        f"data.test.data_root={data_root}",
        f"data.test.split={args.split}",
    ]
    if not args.enable_flash:
        options.append("model.backbone.enable_flash=false")

    cmd = [
        "python",
        str(tools_test),
        "--config-file",
        str(args.config_file.resolve()),
        "--num-gpus",
        str(args.num_gpus),
        "--options",
        *options,
    ]

    print(" ".join(cmd))
    if args.dry_run:
        return

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Pointcept inference failed (exit {result.returncode})")
    print(f"Done. Outputs: {args.save_path.resolve()}")


if __name__ == "__main__":
    main()
