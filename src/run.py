"""Unified experiment runner for Track E CP3."""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import load_data
from src.eval_protocol import (
    build_seen_items,
    build_relevant_items,
    build_activity_groups,
    get_evaluable_users,
    validate_recommendations,
)
from src.metrics import evaluate, summarize_metrics

from src.popularity import PopularityRecommender
from src.mf_als import ALSRecommender
from src.tfidf_cbf import TfidfCBFRecommender
from src.embedding_cbf import (
    EmbeddingCBFRecommender,
    sentence_transformer_encoder,
)


def load_config(config_path):
    """Load experiment configuration from YAML."""

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def get_git_commit():
    """Get current Git commit hash."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def create_model(model_name, config, data):
    """Initialize and fit the selected recommendation model."""

    model_config = config.get("model", {})

    if model_name == "popularity":

        model = PopularityRecommender()
        model.fit(data["train"], data["candidates"])

    elif model_name == "als":

        model = ALSRecommender(
            factors=model_config.get("factors", 64),
            reg=model_config.get("reg", 0.1),
            alpha=model_config.get("alpha", 10.0),
            iterations=model_config.get("iterations", 15),
            seed=config["evaluation"]["seed"],
            verbose=True,
        )

        model.fit(data["train"], data["candidates"])

    elif model_name == "tfidf":

        model = TfidfCBFRecommender(
            ngram_range=tuple(
                model_config.get("ngram_range", [1, 1])
            ),
            min_df=model_config.get("min_df", 2),
            max_df=model_config.get("max_df", 0.5),
            max_features=model_config.get("max_features"),
            sublinear_tf=model_config.get("sublinear_tf", True),
            stop_words=model_config.get("stop_words", "english"),
        )

        model.fit(
            data["train"],
            data["candidates"],
            data["items"],
        )

    elif model_name == "embedding":

        encoder = sentence_transformer_encoder(
            model_config.get(
                "model_name",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            batch_size=model_config.get("batch_size", 64),
            max_seq_length=model_config.get("max_seq_length", 256),
            device=model_config.get("device", "cpu"),
        )

        short_name = model_config.get(
            "model_name",
            "sentence-transformers/all-MiniLM-L6-v2",
        ).split("/")[-1]

        model = EmbeddingCBFRecommender(
            encoder=encoder,
            cache_name=f"{short_name}_len{encoder.max_seq_length}",
        )

        model.fit(
            data["train"],
            data["candidates"],
            data["items"],
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return model


def generate_recommendations(
    model,
    model_name,
    users,
    seen,
    k,
):
    """Generate Top-K recommendations for all evaluation users."""

    if model_name == "popularity":

        return {
            user: model.recommend(
                user,
                seen.get(user, set()),
                k,
            )
            for user in users
        }

    return model.recommend_many(
        users,
        seen,
        k=k,
    )


def run_experiment(config_path, subset_override=None):
    """Execute one experiment and save its results."""

    config = load_config(config_path)

    experiment = config["experiment"]
    evaluation = config["evaluation"]

    model_name = experiment["model"]
    experiment_id = experiment["id"]

    k = evaluation["k"]
    seed = evaluation["seed"]

    subset = (
        subset_override
        if subset_override is not None
        else evaluation.get("subset", 0)
    )

    data_dir = config["data"]["directory"]
    output_dir = Path(config["output"]["directory"])

    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    print("\n====================================")
    print(f"Experiment: {experiment_id}")
    print(f"Model: {model_name}")
    print("====================================")

    start_time = time.perf_counter()

    # 1. Load official training and validation data.
    data = load_data(data_dir)

    train = data["train"]
    candidates = data["candidates"]

    # 2. Build evaluation protocol.
    seen = build_seen_items(train)

    users = get_evaluable_users(
        train,
        data["val_warm"],
        data["eval_users"],
    )

    if subset > 0 and subset < len(users):

        rng = np.random.default_rng(seed)

        users = sorted(
            rng.choice(
                users,
                size=subset,
                replace=False,
            ).tolist()
        )

    relevant = build_relevant_items(
        data["val_warm"],
        users,
    )

    activity_groups = build_activity_groups(
        train,
        users,
    )

    print(f"Evaluable users: {len(users)}")
    print(f"Candidate items: {len(candidates)}")

    # 3. Fit model.
    model = create_model(
        model_name,
        config,
        data,
    )

    # 4. Generate recommendations.
    recommendations = generate_recommendations(
        model,
        model_name,
        users,
        seen,
        k,
    )

    # 5. Validate recommendation protocol.
    validate_recommendations(
        recommendations,
        candidates,
        seen,
        k,
    )

    # 6. Calculate per-user metrics.
    per_user = evaluate(
        recommendations,
        relevant,
        k,
    )

    overall = summarize_metrics(per_user)

    # 7. Calculate metrics by activity group.
    group_labels = activity_groups.reindex(
        per_user.index
    )

    per_user["activity_group"] = group_labels

    by_group = (
        per_user.groupby(
            "activity_group",
            observed=False,
        )
        .agg(
            n_users=("Recall@10", "count"),
            **{
                "Recall@10": ("Recall@10", "mean"),
                "NDCG@10": ("NDCG@10", "mean"),
            },
        )
    )

    # 8. Calculate runtime.
    runtime = time.perf_counter() - start_time

    # 9. Save result files.
    run_name = (
        f"{experiment_id}_{model_name}"
        f"_subset{subset or 'all'}"
    )

    per_user_path = output_dir / f"{run_name}_per_user.csv"
    group_path = output_dir / f"{run_name}_by_group.csv"
    overall_path = output_dir / f"{run_name}_overall.json"

    per_user.to_csv(per_user_path)
    by_group.to_csv(group_path)

    overall_result = {
        **overall,
        "experiment_id": experiment_id,
        "model": model_name,
        "n_users": len(users),
        "n_candidates": len(candidates),
        "runtime_seconds": round(runtime, 3),
    }

    with open(overall_path, "w", encoding="utf-8") as file:
        json.dump(
            overall_result,
            file,
            indent=2,
        )

    # 10. Create configuration hash.
    config_string = json.dumps(
        config,
        sort_keys=True,
        default=str,
    )

    config_hash = hashlib.sha256(
        config_string.encode("utf-8")
    ).hexdigest()

    # 11. Record experiment metadata.
    log_entry = {
        "run_id": run_name,
        "experiment_id": experiment_id,
        "model": model_name,
        "config_hash": config_hash,
        "git_commit": get_git_commit(),
        "n_evaluable_users": len(users),
        "recall@10": overall.get("Recall@10"),
        "ndcg@10": overall.get("NDCG@10"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(runtime, 3),
        "seed": seed,
        "subset": subset,
    }

    log_path = output_dir / "runs.jsonl"

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(log_entry) + "\n"
        )

    # 12. Print summary.
    print("\nExperiment results:")
    print(json.dumps(overall_result, indent=2))

    print("\nResults by activity group:")
    print(by_group)

    print(f"\nRuntime: {runtime:.2f} seconds")
    print(f"Saved log: {log_path}")

    return overall_result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Track E CP3 Experiment Runner"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML configuration",
    )

    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Override evaluation subset (0 = all users)",
    )

    args = parser.parse_args()

    run_experiment(
        args.config,
        args.subset,
    )