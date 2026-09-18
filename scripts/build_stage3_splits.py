"""Build Stage-3 subtype splits gated by macula status.

Cascade:
  Stage1 RD → Stage2 macula intact/detached → Stage3:
    · Macula_Intact  → TD vs ND
    · Macula_Detached → TD vs Bilateral

WARNING: In the current ERDES folder layout each subtype directory is dominated
by a single patient ID, so video-level stratified splits will leak identity.
Results are for local prototyping / explainability templates only until more
patients are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "macula_intact_td_vs_nd": {
        # 1 = TD (total-like), 0 = ND (non-total / localized-like)
        0: ["Retinal_Detachment/Macula_Intact/ND"],
        1: ["Retinal_Detachment/Macula_Intact/TD"],
        "label_names": {0: "ND", 1: "TD"},
        "gate": "macula_intact",
    },
    "macula_detached_td_vs_bilateral": {
        # 1 = TD, 0 = Bilateral
        0: ["Retinal_Detachment/Macula_Detached/Bilateral"],
        1: ["Retinal_Detachment/Macula_Detached/TD"],
        "label_names": {0: "Bilateral", 1: "TD"},
        "gate": "macula_detached",
    },
}


def patient_id(path: str) -> str:
    stem = Path(path).stem
    return stem.split("_")[0]


def collect(data_root: Path, rel_dirs: list[str], label: int) -> list[dict]:
    rows = []
    for rel in rel_dirs:
        folder = data_root / rel
        if not folder.is_dir():
            print(f"[!] missing dir: {folder}")
            continue
        for mp4 in sorted(folder.glob("*.mp4")):
            rel_path = mp4.relative_to(data_root).as_posix()
            rows.append(
                {
                    "path": rel_path,
                    "label": label,
                    "patient_id": patient_id(mp4.name),
                    "subtype": Path(rel).name,
                }
            )
    return rows


def split_df(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=0.10, stratify=train_val["label"], random_state=seed
    )
    return train, val, test


def summarize(df: pd.DataFrame, name: str) -> None:
    print(f"  {name}: n={len(df)} labels={df['label'].value_counts().to_dict()} "
          f"patients={sorted(df['patient_id'].unique().tolist())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage-3 macula-gated subtype CSVs")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "erdes")
    parser.add_argument("--out-root", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = args.data_root
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")

    for task, cfg in TASKS.items():
        rows = []
        for label, dirs in cfg.items():
            if not isinstance(label, int):
                continue
            rows.extend(collect(data_root, dirs, label))
        if not rows:
            print(f"[!] no videos for {task}, skip")
            continue
        df = pd.DataFrame(rows)
        # unique patients per class — emit strong warning
        for lab, sub in df.groupby("label"):
            n_pat = sub["patient_id"].nunique()
            if n_pat <= 1:
                print(
                    f"[WARN] {task} label={lab} ({cfg['label_names'][lab]}): "
                    f"only {n_pat} patient id(s) — identity leakage likely."
                )

        train, val, test = split_df(df, seed=args.seed)
        out_dir = args.out_root / task
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_df_ in ("train", train), ("val", val), ("test", test):
            cols = ["path", "label"]
            split_df_[cols].to_csv(out_dir / f"{split_name}.csv", index=False)
        print(f"[OK] {task}")
        summarize(train, "train")
        summarize(val, "val")
        summarize(test, "test")
        print(f"    saved -> {out_dir}")
        import json

        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "gate": cfg["gate"],
                    "label_names": {str(k): v for k, v in cfg["label_names"].items()},
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                    "patients": sorted(df["patient_id"].unique().tolist()),
                    "warning": (
                        "Single-patient-per-class in current tree; "
                        "do not overclaim generalization."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
