import json
from pathlib import Path
import torch

from src.oil_dataset_plot_core import plot_oil_dataset_gp_lvm_results
from src.evaluate_gp_metrics import evaluate_gp_lvm_model_metrics, save_metrics_json


def tensor_dict_to_json(d):
    return {
        k: (v.tolist() if torch.is_tensor(v) else v)
        for k, v in d.items()
    }


def save_snapshot(snap, labels, fractions, Y, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    snap_json = tensor_dict_to_json(snap)
    with open(save_dir / "snapshot_model.json", "w") as f:
        json.dump(snap_json, f, indent=2)

    plot_oil_dataset_gp_lvm_results(snap, labels, fractions, save_dir)

    snap_metrics = evaluate_gp_lvm_model_metrics(snap, Y)
    save_metrics_json(snap_metrics, save_dir / "snapshot_metrics.json")


def save_final_results(results_dict, labels, fractions, metrics, config_name, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    final_json_keys = [
        "mu_x", "log_s2x", "Z", "log_sf2",
        "log_alpha", "m_u", "C_u"
    ]
    final_json = tensor_dict_to_json({k: results_dict[k] for k in final_json_keys})
    with open(save_dir / "final_model_result.json", "w") as f:
        json.dump(final_json, f, indent=2)

    torch.save(results_dict["predictive_mean"], save_dir / "predictive_mean.pt")
    torch.save(results_dict["predictive_variance"], save_dir / "predictive_variance.pt")

    plot_oil_dataset_gp_lvm_results(results_dict, labels, fractions, save_dir)

    metrics_path = save_dir / f"{config_name}_metrics.json"
    save_metrics_json(metrics, metrics_path)

    torch.save(results_dict, save_dir / "trained_model_dict.pt")


def save_all_results(results_dict, labels, fractions, Y, metrics, config_name, base_save_dir):
    snapshots = results_dict.get("snapshots", {})

    if snapshots:
        for iter_num, snap in snapshots.items():
            snap_dir = base_save_dir / f"{iter_num}_iters_results"
            save_snapshot(snap, labels, fractions, Y, snap_dir)
    else:
        print("Snapshots dict is empty. Skipping snapshot saving.")

    final_dir = base_save_dir / "final_results"
    save_final_results(results_dict, labels, fractions, metrics, config_name, final_dir)
