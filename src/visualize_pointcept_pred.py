import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from scipy.spatial import cKDTree

matplotlib.use("Agg")

SCANNET_CLASSES = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Color a mesh by Pointcept per-point predictions"
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=repo_root / "data" / "open_pit_mine_with_rotated_with_color.obj",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=repo_root
        / "cache"
        / "pointcept_input"
        / "open_pit_mine_with_rotated_with_color"
        / "val"
        / "open_pit_mine_with_rotated_with_color",
        help="Scene folder written by prepare_pointcept_input.py",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="Prediction npy (default: outputs/pointcept/result/<scene>_pred.npy)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "outputs" / "pointcept" / "visualized",
    )
    return parser.parse_args()


def palette():
    cmap = matplotlib.colormaps["tab20"]
    return np.array(
        [np.array(to_rgba(cmap(i % 20))) * 255 for i in range(len(SCANNET_CLASSES))],
        dtype=np.uint8,
    )


def save_preview(points, colors, labels, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    rgb = colors[:, :3] / 255.0
    axes[0].scatter(points[:, 0], points[:, 1], c=rgb, s=1.5, linewidths=0)
    axes[0].set_title("top view (XY)")
    axes[0].set_aspect("equal")
    axes[1].scatter(points[:, 0], points[:, 2], c=rgb, s=1.5, linewidths=0)
    axes[1].set_title("side view (XZ)")
    axes[1].set_aspect("equal")
    present = np.unique(labels)
    handles = [
        Patch(facecolor=colors[labels == i][0] / 255.0, label=SCANNET_CLASSES[i])
        for i in present
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    scene_name = scene_dir.name
    repo_root = Path(__file__).resolve().parents[1]

    pred_path = args.pred
    if pred_path is None:
        pred_path = (
            repo_root / "outputs" / "pointcept" / "result" / f"{scene_name}_pred.npy"
        )
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction not found: {pred_path}")

    pred = np.load(pred_path).reshape(-1)
    coord = np.load(scene_dir / "coord.npy")
    meta = json.loads((scene_dir / "meta.json").read_text())
    points = coord / meta["scale"] + np.asarray(meta["center"], dtype=np.float32)

    if pred.shape[0] != points.shape[0]:
        raise ValueError(
            f"Prediction has {pred.shape[0]} labels but scene has {points.shape[0]} points"
        )

    mesh = trimesh.load_mesh(str(args.mesh.resolve()), force="mesh")
    tree = cKDTree(points)
    dist, nn = tree.query(mesh.triangles_center, workers=-1)
    face_labels = pred[nn].astype(np.int32)

    colors = palette()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, face_colors=colors[face_labels]
    )
    glb_path = args.out_dir / f"{scene_name}_pred.glb"
    mesh.export(str(glb_path))
    np.save(args.out_dir / f"{scene_name}_face_labels.npy", face_labels)
    save_preview(
        points, colors[pred], pred, args.out_dir / f"{scene_name}_pred_points.png"
    )

    print(f"Nearest sampled point is up to {dist.max():.2f} m from a face centroid")
    print(f"Wrote {glb_path}")
    ids, counts = np.unique(face_labels, return_counts=True)
    for i, n in sorted(zip(ids, counts), key=lambda x: -x[1]):
        print(f"{SCANNET_CLASSES[i]:>16} {n:8d} faces {100 * n / face_labels.size:5.1f}%")


if __name__ == "__main__":
    main()
