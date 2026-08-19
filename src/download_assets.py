import argparse
from pathlib import Path

import gdown
import urllib.request


SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
SAMESH_DATASET_DRIVE_ID = "1qzxZZ-RUShNgUKXBPnpI1-Mlr8MkWekN"
POINTCEPT_URL = (
    "https://huggingface.co/Pointcept/PointTransformerV3/resolve/main/"
    "scannet-semseg-pt-v3m1-0-base/model/model_best.pth"
)


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Already present: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")


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
    parser = argparse.ArgumentParser(description="Download SAMesh and Pointcept checkpoints")
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=repo_root / "checkpoints" / "samesh" / "sam2_hiera_large.pt",
    )
    parser.add_argument(
        "--pointcept-checkpoint",
        type=Path,
        default=repo_root / "checkpoints" / "pointcept" / "model_best.pth",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=repo_root / "datasets",
    )
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-sam2", action="store_true")
    parser.add_argument("--skip-pointcept", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.skip_sam2:
        download_file(SAM2_URL, args.sam2_checkpoint.resolve())
    if not args.skip_pointcept:
        download_file(POINTCEPT_URL, args.pointcept_checkpoint.resolve())
    if not args.skip_dataset:
        download_samesh_dataset(args.datasets_dir.resolve())


if __name__ == "__main__":
    main()
