import os
import shutil

from src.core.config import load_config
from src.core.seed import set_global_seed
from src.core.logging import make_run_dir, save_metrics

from src.data.loader import load_hr_analytics
from src.models.baseline_registry import get_baseline_models
from src.evaluation.crossval import run_cv
from src.evaluation.aggregate import aggregate_cv
from src.plotting.plots import plot_fairness_bars, plot_fairness_vs_auc


def main(config_path: str):
    # Load config
    config = load_config(config_path)

    seed = config["seed"]
    set_global_seed(seed)

    # Load dataset
    ds = load_hr_analytics(config["dataset_path"])

    # Build models
    models = get_baseline_models(ds.X, seed)

    # Run CV
    records = run_cv(
        models=models,
        X=ds.X,
        y=ds.y,
        A=ds.A,
        seed=seed,
        n_splits=config["cv_folds"],
    )

    # Create run directory
    run_dir = make_run_dir()

    # Save metrics
    save_metrics(run_dir, records)

    # Aggregate CV results
    summary = aggregate_cv(records)
    summary.to_csv(os.path.join(run_dir, "cv_summary.csv"), index=False)

    # Save plots
    plot_dir = os.path.join(run_dir, "plots")
    plot_fairness_bars(summary, plot_dir)
    plot_fairness_vs_auc(summary, plot_dir)

    # Copy config snapshot
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))

    print("Run complete.")
    print("Outputs saved to:", run_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)

