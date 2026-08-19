import argparse
from pathlib import Path

import matplotlib
import trimesh.visual.color
from omegaconf import OmegaConf
from samesh.models.sam_mesh import segment_mesh


def patch_trimesh_colormaps():
    """trimesh >=4.5 only ships four colormaps; SAMesh still asks for 'jet'."""
    original = trimesh.visual.color.interpolate

    def interpolate(values, color_map=None, **kwargs):
        if isinstance(color_map, str) and color_map in matplotlib.colormaps:
            color_map = matplotlib.colormaps[color_map]
        return original(values, color_map=color_map, **kwargs)

    trimesh.visual.color.interpolate = interpolate
    trimesh.visual.interpolate = interpolate


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run SAMesh auto mesh segmentation")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=repo_root / "data" / "open_pit_mine_with_rotated_with_color.obj",
        help="Path to input mesh (.obj / .glb / .ply)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs" / "mesh_segmentation.yaml",
        help="Path to SAMesh config yaml",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write intermediate visualization caches",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mesh_path = args.mesh.resolve()
    config_path = args.config.resolve()

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    patch_trimesh_colormaps()

    config = OmegaConf.load(str(config_path))
    print(f"Segmenting mesh with SAMesh: {mesh_path}")
    mesh = segment_mesh(str(mesh_path), config, visualize=args.visualize)
    print(f"Done. Output dir: {config.output}")
    return mesh


if __name__ == "__main__":
    main()
