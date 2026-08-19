# 3D SAM AI Test (SAMesh)

Auto mesh part segmentation with Segment Any Mesh (SAM2).

## Setup

Fetch samesh codes as submodule:
```bash
git submodule update --init --recursive
```

Run installation using UV
```
uv sync
uv pip install -e third_party/samesh
uv pip install setuptools wheel
uv pip install --no-build-isolation -e third_party/samesh/third_party/segment-anything-2
uv run python src/download_assets.py
```

## Run

```bash
uv run python src/segment_mesh.py \
  --mesh datasets/backflip-benchmark-remeshed-processed/axe.glb \
  --visualize
```

Open-pit mesh:

```bash
uv run python src/segment_mesh.py \
  --mesh data/open_pit_mine_with_rotated_with_color.obj \
  --visualize
```

Outputs: `outputs/mesh_segmentation`, `cache/mesh_segmentation`

## Notes

`third_party/samesh` is a git submodule (includes Meta SAM2 as a nested submodule). Do not vendor it as plain files.
