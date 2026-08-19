import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Sample a mesh into a ScanNet-style Pointcept dataset folder"
    )
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
        help="Dataset root (default: cache/pointcept_input/<mesh_stem>)",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Split folder name the Pointcept dataset will read",
    )
    parser.add_argument(
        "--scene-name",
        default=None,
        help="Scene folder name (default: mesh stem)",
    )
    parser.add_argument(
        "--target-extent",
        type=float,
        default=10.0,
        help="Rescale so the largest axis extent equals this many metres; 0 keeps world scale",
    )
    return parser.parse_args()


def sample_colors(mesh, face_ids):
    visual = mesh.visual
    try:
        if hasattr(visual, "to_color"):
            visual = visual.to_color()
        return np.asarray(visual.face_colors, dtype=np.float32)[face_ids, :3]
    except Exception:
        return np.full((len(face_ids), 3), 128.0, dtype=np.float32)


def main():
    args = parse_args()
    mesh_path = args.mesh.resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.out_dir
    if dataset_root is None:
        dataset_root = repo_root / "cache" / "pointcept_input" / mesh_path.stem
    scene_name = args.scene_name or mesh_path.stem
    scene_dir = dataset_root / args.split / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
    if mesh.is_empty:
        raise RuntimeError(f"Failed to load mesh: {mesh_path}")

    points, face_ids = trimesh.sample.sample_surface_even(mesh, args.num_points)
    points = np.asarray(points, dtype=np.float32)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    colors = sample_colors(mesh, face_ids)
    normals = np.asarray(mesh.face_normals, dtype=np.float32)[face_ids]

    center = points.mean(axis=0)
    scale = 1.0
    extent = float(np.max(points.max(axis=0) - points.min(axis=0)))
    if args.target_extent > 0 and extent > 0:
        scale = args.target_extent / extent
    coord = ((points - center) * scale).astype(np.float32)

    np.save(scene_dir / "coord.npy", coord)
    np.save(scene_dir / "color.npy", colors.astype(np.float32))
    np.save(scene_dir / "normal.npy", normals)
    np.save(scene_dir / "face_ids.npy", face_ids.astype(np.int32))

    meta = {
        "mesh": str(mesh_path),
        "num_points": int(coord.shape[0]),
        "world_extent": extent,
        "center": center.tolist(),
        "scale": scale,
    }
    (scene_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote Pointcept dataset to {dataset_root}")
    print(f"scene: {args.split}/{scene_name}, coord.npy shape={coord.shape}")
    print(f"world extent {extent:.2f} m scaled by {scale:.6f}")


if __name__ == "__main__":
    main()
