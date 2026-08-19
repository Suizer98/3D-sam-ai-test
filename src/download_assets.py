import argparse
from pathlib import Path

import gdown
import urllib.request


SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
SAMESH_DATASET_DRIVE_ID = "1qzxZZ-RUShNgUKXBPnpI1-Mlr8MkWekN"


def download_sam2_checkpoint(checkpoint_path: Path):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 1_000_000:
        print(f"SAM2 checkpoint already present: {checkpoint_path}")
        return
    print(f"Downloading SAM2 checkpoint to {checkpoint_path}")
    urllib.request.urlretrieve(SAM2_URL, checkpoint_path)
    print(f"Saved {checkpoint_path} ({checkpoint_path.stat().st_size} bytes)")


def download_samesh_dataset(datasets_dir: Path):
    datasets_dir.mkdir(parents=True, exist_ok=True)
    zip_path = datasets_dir / "samesh_curated.zip"
    extract_marker = datasets_dir / "backflip-benchmark-remeshed-processed"
    if extract_marker.exists():
        mesh_count = len(list(extract_marker.glob("*.glb")))
        print(f"Dataset folder already present: {extract_marker} ({mesh_count} glb)")
        return
    if not zip_path.exists():
        url = f"https://drive.google.com/uc?id={SAMESH_DATASET_DRIVE_ID}"
        print(f"Downloading SAMesh curated dataset to {zip_path}")
        gdown.download(url, str(zip_path), quiet=False)
    print(f"Extracting {zip_path} -> {datasets_dir}")
    gdown.extractall(str(zip_path), str(datasets_dir))
    if extract_marker.exists():
        mesh_count = len(list(extract_marker.glob("*.glb")))
        print(f"Dataset ready: {extract_marker} ({mesh_count} glb)")
    else:
        print("Extract finished. Check datasets/ for extracted mesh folders.")


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download SAMesh checkpoint and dataset")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "checkpoints" / "sam2_hiera_large.pt",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=repo_root / "datasets",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Only download SAM2 checkpoint",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Only download curated dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.skip_checkpoint:
        download_sam2_checkpoint(args.checkpoint.resolve())
    if not args.skip_dataset:
        download_samesh_dataset(args.datasets_dir.resolve())


if __name__ == "__main__":
    main()
