# 3D SAM AI Test

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

Checkpoints:

- `checkpoints/samesh/sam2_hiera_large.pt`
- `checkpoints/pointcept/model_best.pth` (ScanNet PTv3)

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
uv pip install torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
cd ../third_party/pointcept/libs/pointops
uv run --project ~/3D-sam-ai-test/pointcept python setup.py install
```

`pointops` needs a real CUDA compiler (`nvcc`). WSL2 GPU drivers are not enough, and `nvidia-cuda-nvcc-cu12` pip wheels often ship only `ptxas`. Install the toolkit, then rebuild from `libs/pointops`:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit
which nvcc
export CUDA_HOME=/usr
export TORCH_CUDA_ARCH_LIST=8.9
uv run --project ~/3D-sam-ai-test/pointcept python setup.py install
```

PTv3 configs may run without `pointops` (that lib is mainly PTv1/v2). Skip the setup.py step if you only use PT-v3.

Do not run `uv pip install` until `uv sync` has created `pointcept/.venv`.
Do not run `python setup.py` from inside `pointcept/` — that folder has no `setup.py`.
Do not add `torch-cluster` (or other PyG source packages) to `pointcept/pyproject.toml`. Install the PyG wheels with `uv pip` as above. After that, use `uv run --no-sync` so uv does not try to compile them from source (`No module named 'torch'` in build isolation).

Pointcept's import chain also needs `peft`, `transformers==4.50.3`, `wandb`, `torch_geometric` and `open3d`. They are in `pointcept/pyproject.toml`, so `uv sync` covers them.

Prepare points from the mesh (does not overwrite the OBJ):

```bash
uv run --no-sync --project pointcept python src/prepare_pointcept_input.py \
  --mesh data/open_pit_mine_with_rotated_with_color.obj \
  --num-points 50000
```

This writes a ScanNet-style dataset, since the ScanNet config reads
`<root>/<split>/<scene>/{coord,color,normal}.npy`:

```
cache/pointcept_input/open_pit_mine_with_rotated_with_color/
  val/open_pit_mine_with_rotated_with_color/
    coord.npy color.npy normal.npy face_ids.npy meta.json
```

The mine is ~512 m across but ScanNet PTv3 voxelizes at 0.02 m, so points are centred and
rescaled to `--target-extent` (10 m default). `meta.json` keeps the centre and scale, and
`face_ids.npy` maps each point back to its source mesh face. Pass `--target-extent 0` to keep
world coordinates.

Inference (ScanNet labels will not match mine classes):

```bash
uv run --no-sync --project pointcept python src/run_pointcept_infer.py \
  --config-file third_party/pointcept/configs/scannet/semseg-pt-v3m1-0-base.py \
  --checkpoint checkpoints/pointcept/model_best.pth \
  --input-root cache/pointcept_input/open_pit_mine_with_rotated_with_color \
  --save-path outputs/pointcept
```

The runner disables flash attention by default (`model.backbone.enable_flash=false`), because
`flash_attn` is not installed and building it is slow. Pass `--enable-flash` if you install it.
The runner also has to override `data.test.data_root`, not just `data_root`: the config copies
`data_root` into the nested dataset dicts when the file is executed, so `--options data_root=...`
alone leaves the test split pointing at `data/scannet`.

Color the mesh by the predictions:

```bash
uv run --no-sync --project pointcept python src/visualize_pointcept_pred.py
```

This un-scales the sampled points back to world coordinates using `meta.json`, assigns each mesh
face the label of its nearest sampled point, and writes to `outputs/pointcept/visualized/`:
a `_pred.glb` with per-face colors, a `_face_labels.npy`, and a `_pred_points.png` top/side preview.
Mesh detail is limited by `--num-points`; at 50k points over 512 m the nearest sample can be
almost 6 m from a face centroid.

Predictions land in `outputs/pointcept/result/<scene>_pred.npy` (one ScanNet class id per point)
plus a `submit/<scene>.txt`. Reported mIoU/mAcc are 0 or NaN because there is no ground truth;
that is expected. On the mine mesh the model calls roughly 43% wall, 39% floor and 17% sofa,
which confirms the pipeline runs but that indoor classes carry no mining meaning.

## Notes

`third_party/samesh` and `third_party/pointcept` are git submodules (code).
`samesh/` and `pointcept/` are only the uv environments.
`third_party/samesh` includes Meta SAM2 as a nested submodule. Do not vendor these as plain files.
