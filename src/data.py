"""Data loading utilities for Track E CP3."""

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {
    "train": {"user_id", "item_id", "rating", "timestamp"},
    "candidates": {"item_id"},
    "val_warm": {"user_id", "item_id", "rating", "timestamp"},
    "eval_users": {"user_id"},
    "items": {"item_id", "content_text_basic"},
}


def load_data(data_dir="."):
    """
    Load official training and warm-validation data.

    The test split is intentionally not loaded.
    """
    root = Path(data_dir)

    main_dir = root / "MAIN DATA"
    supporting_dir = root / "SUPPORTING_DATA"

    paths = {
        "train": main_dir / "interactions_train.csv",
        "candidates": main_dir / "candidate_items_warm.csv",
        "val_warm": supporting_dir / "interactions_validation_warm.csv",
        "eval_users": supporting_dir / "evaluable_users_validation_warm.csv",
        "items": main_dir / "items.csv",
    }

    missing = [str(path) for path in paths.values() if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Required dataset files are missing:\n" + "\n".join(missing)
        )

    data = {
        "train": pd.read_csv(paths["train"]),
        "candidates": pd.read_csv(paths["candidates"])["item_id"].tolist(),
        "val_warm": pd.read_csv(paths["val_warm"]),
        "eval_users": pd.read_csv(paths["eval_users"])["user_id"].tolist(),
        "items": pd.read_csv(paths["items"]),
    }

    for name, required in REQUIRED_COLUMNS.items():
        if name == "candidates":
            actual = {"item_id"} if data[name] else set()
        elif name == "eval_users":
            actual = {"user_id"} if data[name] else set()
        else:
            actual = set(data[name].columns)

        missing_columns = required - actual

        if missing_columns:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing_columns)}"
            )

    return data