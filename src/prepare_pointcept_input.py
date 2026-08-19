import argparse
from pathlib import Path

import numpy as np
import trimesh


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sample a mesh into Pointcept npy inputs")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=repo_root / "data" / "open_pit_mine_with_rotated_with_color.obj",
        help="Input mesh (.obj / .glb / .ply)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=50000,
        help="Number of surface samples",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: cache/pointcept_input/<mesh_stem>)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mesh_path = args.mesh.resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    out_dir = args.out_dir
    if out_dir is None:
        repo_root = Path(__file__).resolve().parents[1]
        out_dir = repo_root / "cache" / "pointcept_input" / mesh_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
    if mesh.is_empty:
        raise RuntimeError(f"Failed to load mesh: {mesh_path}")

    points, face_ids = trimesh.sample.sample_surface_even(mesh, args.num_points)
    points = np.asarray(points, dtype=np.float32)
    strength = np.ones((points.shape[0],), dtype=np.float32)
    segment = np.zeros((points.shape[0],), dtype=np.int32)

    np.save(out_dir / "coord.npy", points)
    np.save(out_dir / "strength.npy", strength)
    np.save(out_dir / "segment.npy", segment)
    np.save(out_dir / "face_ids.npy", np.asarray(face_ids, dtype=np.int32))

    print(f"Wrote Pointcept input to {out_dir}")
    print(f"coord.npy shape={points.shape}")


if __name__ == "__main__":
    main()
