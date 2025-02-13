import argparse
import json
import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import yaml

metric_map = {
    "charts/rolling_mean_return": "mean_episodic_return",
    "charts/rolling_success_rate": "mean_episodic_success",
    "charts/episodic_length": "mean_episodic_length",
}
config_map = {
    "env_id": "env",
    "seed": "seed",
    "total_timesteps": "total_timesteps",
    "hidden_size": "hidden_size",
    "hidden_layers": "hidden_layers",
    "backbone": "backbone",
    "backbone_variant": "backbone_variant",
    "freeze_param_filter": "freeze_param_filter",
    "checkpoint_load_path": "checkpoint_load_path",
    "checkpoint_param_filters": "checkpoint_param_filter",
    "use_checkpoint_images": "use_checkpoint_images",
    "early_stop_patience": "early_stop_patience",
    "use_reward_model": "use_reward_model",
    "use_reward_loss": "use_reward_loss",
    "use_transition_model": "use_transition_model",
    "use_transition_loss": "use_transition_loss",
    "use_contrastive_loss": "use_curl_loss",
    "use_curl_loss": "use_curl_loss",
    "use_reconstruction_loss": "use_reconstruction_loss",
    "use_data_augmentation_loss": "use_data_augmentation_loss",
    "use_dbc_loss": "use_dbc_loss",
    "use_spr_loss": "use_spr_loss",
    "apply_data_augmentation": "apply_data_augmentation",
    "augmentations": "augmentations",
    "contrastive_temperature": "contrastive_temperature",
    "contrastive_positives": "contrastive_positives",
    "variational_reconstruction": "variational_reconstruction",
    "polyak_tau": "polyak_tau",
    "spr_loss_coef": "spr_loss_coef",
    "probabilistic_transition_model": "probabilistic_transition_model",
    "min_transition_model_sigma": "min_transition_model_sigma",
    "max_transition_model_sigma": "max_transition_model_sigma",
    "encoder_dropout": "encoder_dropout",
    "repr_eval_every": "repr_eval_every",
    "optimizer": "optimizer",
    # sac
    "tau": "tau",
    "autotune": "autotune",
    "alpha_lr": "alpha_lr",
    "alpha_beta": "alpha_beta",
    "alpha": "alpha",
    "independent_encoder": "independent_encoder",
    "encoder_tau": "encoder_tau",
}


def extract_metrics_from_tensorboard(run_folder, output_folder):
    ea = event_accumulator.EventAccumulator(run_folder)
    ea.Reload()

    data = {metric: {} for metric in metric_map.values()}
    data["steps"] = set()

    for metric in metric_map.keys():
        try:
            events = ea.Scalars(metric)
        except KeyError as e:
            print("Available metrics:", ea.Tags()["scalars"])
            raise e
        for event in events:
            data[metric_map[metric]][event.step] = event.value
            data["steps"].add(event.step)

    # Convert the data to a DataFrame
    all_steps = sorted(data["steps"])
    df_data = {"steps": all_steps}
    for metric in metric_map.values():
        df_data[metric] = [data[metric].get(step, float("nan")) for step in all_steps]

    df = pd.DataFrame(df_data)
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

        if "backbone" not in filtered_config:
            filtered_config["backbone"] = "conv"

        script_name = os.path.basename(run_folder).split("-")[1]
        if script_name.startswith("train_sac"):
            filtered_config["algorithm"] = "SAC"
        elif script_name.startswith("ppo"):
            filtered_config["algorithm"] = "PPO"
        else:
            filtered_config["algorithm"] = script_name.split("_")[0].upper()

        if type(filtered_config.get("checkpoint_param_filter", {})) == dict:
            filtered_config["checkpoint_param_filter"] = filtered_config["checkpoint_param_filter"].get("agent")
        
        if "env_configs" in config and config["env_configs"]:
            if type(config["env_configs"]) == str:
                env_configs = json.loads(config["env_configs"])
            else:
                env_configs = config["env_configs"]

            for key, value in env_configs.items():
                filtered_config[f"env__{key}"] = value

            if "slidingpuzzle" in filtered_config["env"].lower():
                filtered_config["env__size"] = env_configs["w"]

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
    check_wandb,
):
    import wandb

    api = wandb.Api()

    for experiment in tqdm(os.listdir(runs_folder)):
        # Skip experiments before min_datetime
        if experiment < min_datetime or experiment > max_datetime:
            # print(f"Skipping {experiment}: {experiment < min_datetime} or {experiment > max_datetime}")
            continue
        # print(f"Processing {experiment}")

        experiment_path = os.path.join(runs_folder, experiment)
        if os.path.isdir(experiment_path):
            output_folder = os.path.join(output_base_folder, experiment)
            metrics_file = os.path.join(output_folder, "metrics.csv")
            config_file = os.path.join(output_folder, "config.yaml")

            if (override_metrics or override_configs) and not all and not os.path.exists(output_folder):
                # process only runs that were already processed
                continue

            config_path = os.path.join(experiment_path, "config.yaml")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # if config.get("env__size") != 4:
            #     continue

            # Get wandb run info if it was automatic
            if check_wandb:
                # Check if experiment exists and is running/finished in wandb
                try:
                    runs = api.runs(
                        f"{config['wandb_entity']}/{config['wandb_project_name']}",
                        {"display_name": experiment},
                    )
                    run = next(runs)
                except Exception as e:
                    print(f"Error checking wandb state for {experiment_path}: {e}")
                    continue
                if run.state not in ["running", "finished"]:
                    print(f"Skipping {experiment}: {run.state}")
                    continue

            if not os.path.exists(metrics_file) or override_metrics:
                try:
                    extract_metrics_from_tensorboard(experiment_path, output_folder)
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
        "--runs-folder", type=str, default="runs", help="Folder containing the runs."
    )
    parser.add_argument(
        "--output-base-folder",
        type=str,
        default="../visualization/runs",
        help="Base folder for output.",
    )
    parser.add_argument(
        "--all",
        action=argparse.BooleanOptionalAction,
        help="Process all runs in source folder.",
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
        "--check-wandb",
        action=argparse.BooleanOptionalAction,
        help="Check wandb state for runs.",
        default=True,
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
        args.check_wandb,
    )
