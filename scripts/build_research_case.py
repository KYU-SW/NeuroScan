"""Register one Validation case as a Phase 7-B research case (CLI only; no upload UI).

  PYTHONPATH=backend .venv\\Scripts\\python.exe scripts\\build_research_case.py \\
      --case <validation case key> --alias val-001 [--device auto]

Order: alias check -> split check on the case key string (test refused before any file is
opened, validation only) -> frozen Phase 5A inference in research/.staging -> reproducibility vs
the stored Phase 4 validation prediction -> GT / grid / metric / component / difference /
matching checks -> atomic registration (folder rename + DB rows + local alias map). On any
failure the staging folder is removed and nothing is registered. Output names only the alias.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from neuroscan_ai.research.bundle import ResearchBuildError, build_research_bundle  # noqa: E402
from neuroscan_ai.research.sources import (  # noqa: E402
    Isles2022ResearchAdapter,
    ResearchSourceError,
)
from neuroscan_api.db.session import Database  # noqa: E402
from neuroscan_api.services.research import (  # noqa: E402
    ResearchRegistrationError,
    alias_exists,
    discard_staging,
    register_bundle,
    staging_dir,
)
from neuroscan_api.settings import load_settings  # noqa: E402
from neuroscan_api.storage import Storage  # noqa: E402

DEFAULT_REFERENCE = (
    REPO_ROOT / "results/phase4/25d_baseline_20261024-192617_34d18df104da/predictions"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True, help="validation case key (never printed)")
    ap.add_argument("--alias", required=True, help="public alias, e.g. val-001")
    ap.add_argument("--dataset", default="isles2022", choices=["isles2022"])
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/raw/ISLES2022")
    ap.add_argument("--reference-predictions", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    settings = load_settings()
    storage = Storage(settings.storage_root)
    db = Database.open(settings.database_path)
    db.init_schema()  # creates / upgrades to schema v2 (backup first)
    out: dict[str, object] = {"alias": args.alias}
    try:
        if alias_exists(db, args.alias):
            raise ResearchRegistrationError("alias already registered")
        adapter = Isles2022ResearchAdapter(
            args.data_root, settings.split_file(), args.reference_predictions
        )
        source = adapter.resolve(args.case, args.alias)  # before any file is opened
    except (ResearchSourceError, ResearchRegistrationError) as exc:
        out.update(status="refused", code=getattr(exc, "code", "REFUSED"), message=str(exc))
        print(json.dumps(out, ensure_ascii=False))
        return 3
    staging = staging_dir(storage)
    try:
        manifest = build_research_bundle(source, staging, device=args.device)
        register_bundle(
            db,
            storage,
            staging,
            args.alias,
            {"dataset_id": source.dataset_id, "case_key": source.source_case_key},
        )
    except ResearchBuildError as exc:
        discard_staging(staging)
        details = {k: v for k, v in exc.details.items() if k != "error"}
        out.update(status="failed", code=exc.code, message=str(exc), details=details)
        print(json.dumps(out, ensure_ascii=False, default=str))
        return 4
    except Exception as exc:  # registration / IO failure: nothing partial stays behind
        discard_staging(staging)
        out.update(status="failed", code=type(exc).__name__, message=str(exc)[:300])
        print(json.dumps(out, ensure_ascii=False))
        return 4
    finally:
        db.dispose()
    m = json.loads((storage.area("research") / args.alias / "metrics.json").read_text("utf-8"))
    out.update(
        status="registered",
        reproducibility=manifest["reproducibility"],
        metrics={
            k: m["metrics"][k]
            for k in (
                "has_lesion",
                "dice",
                "tp_voxels",
                "fp_voxels",
                "fn_voxels",
                "component_count_gt",
                "component_count_prediction",
            )
        },
    )
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
