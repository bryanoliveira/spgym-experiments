import argparse
import json
import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import yaml

metric_map = {
    # "episode/score": "mean_episodic_return",
    # "epstats/log_success": "mean_episodic_success",
    # "episode/score": "mean_episodic_return",
    # "epstats/log_success": "mean_episodic_success",
    # "epstats/rolling_success_rate": "mean_episodic_success",

    # new
    "episode/rolling_mean_return": "mean_episodic_return",
    "episode/rolling_success_rate": "mean_episodic_success",
    "episode/length": "mean_episodic_length",
}
config_map = {
    "seed": "seed",
}


def extract_metrics_from_json(run_folder, output_folder):
    data = {metric: [] for metric in metric_map.values()}
    data["steps"] = []

    with open(os.path.join(run_folder, "metrics.jsonl"), "r") as f:
        for line in f:
            metrics = json.loads(line)
            data["steps"].append(metrics["step"])
            for metric, new_metric in metric_map.items():
                value = metrics.get(metric)
                data[new_metric].append(value)

    df = pd.DataFrame(data)
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(os.path.join(output_folder, "metrics.csv"), index=False)


def extract_config(run_folder, output_folder):
    config_path = os.path.join(run_folder, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        filtered_config = {
            config_map[key]: config[key] for key in config_map.keys() if key in config
        }

        filtered_config["algorithm"] = "DreamerV3"
        filtered_config["env"] = {
            "sldp": "SlidingPuzzle-v0",
        }[config["task"].split("_")[0]]
        filtered_config["total_timesteps"] = config["run"]["steps"]
        filtered_config["hidden_size"] = config["enc"]["simple"]["units"]
        filtered_config["hidden_layers"] = config["enc"]["simple"]["layers"]
        filtered_config["backbone"] = "conv"
        filtered_config["backbone_variant"] = config["enc"]["simple"]["depth"]
        filtered_config["disable_decoder"] = (
            config["loss_scales"]["dec_cnn"] == 0
            or config["loss_scales"]["dec_mlp"] == 0
        )

        if filtered_config["env"] == "SlidingPuzzle-v0":
            filtered_config["env__size"] = config["env"]["sldp"]["w"]
            filtered_config["env__variation"] = config["task"].split("_")[1]
            if filtered_config["env__variation"] == "image":
                filtered_config["env__image_folder"] = config["env"]["sldp"][
                    "image_folder"
                ]
                filtered_config["env__image_pool_size"] = config["env"]["sldp"][
                    "image_pool_size"
                ]
                filtered_config["env__image_size"] = config["env"]["sldp"][
                    "image_size"
                ][0]

        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, "config.yaml"), "w") as file:
            yaml.dump(filtered_config, file)


def process_all_experiments(
    runs_folder,
    output_base_folder,
    override_metrics,
    override_configs,
    min_datetime,
    max_datetime,
    all,
):
    for experiment in tqdm(os.listdir(runs_folder)):
        # Skip experiments before min_datetime
        if experiment < min_datetime or experiment > max_datetime:
            # print(f"Skipping {experiment}: {experiment < min_datetime} or {experiment > max_datetime}")
            continue

        experiment_path = os.path.join(runs_folder, experiment)
        output_folder = os.path.join(output_base_folder, experiment)

        if (override_metrics or override_configs) and not all and not os.path.exists(output_folder):
            # process only runs that were already processed
            continue

        if os.path.isdir(experiment_path):
            metrics_file = os.path.join(output_folder, "metrics.csv")
            config_file = os.path.join(output_folder, "config.yaml")
            if not os.path.exists(metrics_file) or override_metrics:
                try:
                    extract_metrics_from_json(experiment_path, output_folder)
                except Exception as e:
                    print(f"Error processing metrics for {experiment_path}: {e}")
            if not os.path.exists(config_file) or override_configs:
                try:
                    extract_config(experiment_path, output_folder)
                except Exception as e:
                    print(f"Error processing config for {experiment_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TensorBoard metrics.")
    parser.add_argument(
        "--runs-folder", type=str, default="logdir", help="Folder containing the runs."
    )
    parser.add_argument(
        "--output-base-folder",
        type=str,
        default="../visualization/runs",
        help="Base folder for output.",
    )
    parser.add_argument(
        "--override",
        action=argparse.BooleanOptionalAction,
        help="Override existing metrics and config files.",
    )
    parser.add_argument(
        "--override-metrics",
        action=argparse.BooleanOptionalAction,
        help="Override existing metrics files.",
    )
    parser.add_argument(
        "--override-configs",
        action=argparse.BooleanOptionalAction,
        help="Override existing config files.",
    )
    parser.add_argument(
        "--min-datetime",
        type=str,
        default="2025",
        help="Minimum datetime to process.",
    )
    parser.add_argument(
        "--max-datetime",
        type=str,
        default="z",
        help="Maximum datetime to process.",
    )
    parser.add_argument(
        "--all",
        action=argparse.BooleanOptionalAction,
        help="Process all runs in source folder.",
    )
    args = parser.parse_args()

    process_all_experiments(
        args.runs_folder,
        args.output_base_folder,
        args.override or args.override_metrics,
        args.override or args.override_configs,
        args.min_datetime,
        args.max_datetime,
        args.all,
    )
