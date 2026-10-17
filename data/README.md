# data/

Local-only. Nothing under this directory except this README and `.gitkeep` files is tracked by Git.

| Folder | Contents |
|---|---|
| `raw/ISLES2022/` | Original ISLES'22 BIDS download, kept unmodified (read-only) |
| `processed/<version>/` | Preprocessed per-patient volumes + `meta.json` (later phase) |
| `splits/` | Patient-wise train/val/test split JSON files (later phase) |

ISLES'22 is distributed under CC BY 4.0 (see `raw/ISLES2022/dataset_description.json` and
`raw/ISLES2022/LICENSE`). Project policy: do not commit raw or processed images to Git.

Cite: Hernandez Petzsche, M.R., de la Rosa, E., Hanning, U. et al. ISLES 2022: A multi-center
magnetic resonance imaging stroke lesion segmentation dataset. *Scientific Data* 9, 762 (2022).
DOI: [10.1038/s41597-022-01875-5](https://doi.org/10.1038/s41597-022-01875-5)
