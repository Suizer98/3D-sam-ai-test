# 3D SAM AI Test (SAMesh)

Auto mesh part segmentation with Segment Any Mesh (SAM2).

## Setup

Fetch submodules (samesh + Pointcept):
```bash
git submodule update --init --recursive
```

SAMesh uv env lives in `samesh/` (isolated from Pointcept):

```bash
cd samesh
uv sync
uv pip install -e ../third_party/samesh
uv pip install setuptools wheel
uv pip install --no-build-isolation -e ../third_party/samesh/third_party/segment-anything-2
cd ..
uv run --project samesh python src/download_assets.py
```

## Run

```bash
uv run --project samesh python src/segment_mesh.py \
  --mesh datasets/backflip-benchmark-remeshed-processed/axe.glb \
  --visualize
```

Open-pit mesh:

```bash
uv run --project samesh python src/segment_mesh.py \
  --mesh data/open_pit_mine_with_rotated_with_color.obj \
  --visualize
```

Outputs: `outputs/mesh_segmentation`, `cache/mesh_segmentation`

## Pointcept (native 3D)

`pointcept/` is only the uv environment. Pointcept code is the submodule `third_party/pointcept`.

After `git submodule update --init --recursive`, from the repo root:

```bash
cd pointcept
uv sync
uv pip install spconv-cu124
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
cd ../third_party/pointcept/libs/pointops
uv run --project ../../../../pointcept python setup.py install
```

The `--project` path is `pointcept/` at the repo root (the uv env), counted from `third_party/pointcept/libs/pointops`. Or use an absolute path: `uv run --project ~/3D-sam-ai-test/pointcept python setup.py install`.

Do not run `uv pip install` until `uv sync` has created `pointcept/.venv`.
Do not run `python setup.py` from inside `pointcept/` — that folder has no `setup.py`.

Prepare points from the mesh (does not overwrite the OBJ):

```bash
uv run --project samesh python src/prepare_pointcept_input.py \
  --mesh data/open_pit_mine_with_rotated_with_color.obj \
  --num-points 50000
```

Inference after you download a Pointcept checkpoint (ScanNet / SemanticKITTI labels will not match mine classes):

```bash
uv run --project pointcept python src/run_pointcept_infer.py \
  --config-file third_party/pointcept/configs/scannet/semseg-pt-v3m1-0-base.py \
  --checkpoint <checkpoint.pth> \
  --input-root cache/pointcept_input/open_pit_mine_with_rotated_with_color \
  --save-path outputs/pointcept
```

## Notes

`third_party/samesh` and `third_party/pointcept` are git submodules (code).
`samesh/` and `pointcept/` are only the uv environments.
`third_party/samesh` includes Meta SAM2 as a nested submodule. Do not vendor these as plain files.
