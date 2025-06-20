import os
import torch
RESOURCE_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../data")
ASSEMBLY_RESOURCE_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../data/assembly")


def update_default_settings(settings: dict,
                            name: str,
                            default: dict):
    sub_settings = settings.get(name, {})
    for item_name, value in default.items():
        if item_name not in sub_settings:
            sub_settings[item_name] = value
    settings[name] = sub_settings
    return sub_settings


default_settings = settings = {
    "assembly": {
        "contact_shrink_ratio": 0.0,
    },
    "env": {
        "n_robot": 2,
        "boundary_part_ids": [0],
        "sim_buffer_size": 512,
        "num_rollouts": 1024,
        "verbose": False,
        "seed": 0,
    },
    "rbe": {
        "nt": 8,
        "mu": 0.55,
        "Ccp": 1E6,
        "density": 1E2,
        "velocity_tol": 1e-2,
        "verbose": False,
    },
    "admm": {
        "sigma": 1E-6,
        "r": 0.1,
        "alpha": 1.6,
        "max_iter": 1000,
        "evaluate_iter": 100,
        "float_type": torch.float32,
    },
    "curriculum": {
        "n_beam": 64,
    },
    "policy": {
        "gat_layers": 8,
        "gat_heads": 1,
        "gat_hidden_dims": 16,
        "gat_dropout": 0.0,
        "centroids": False
    },
    "ppo": {
        "gamma": 0.95,
        "eps_clip": 0.2,

        "base_entropy_weight": 0.005,
        "entropy_weight_increase": 0.001,
        "max_entropy_weight": 0.01,

        "lr_milestones": [100, 300],
        "lr_actor": 2e-3,
        "betas_actor": [0.95, 0.999],

        "per_alpha": 0.8,
        "per_beta": 0.1,
        "per_num_anneal": 500,
    },
    "training": {
        "max_train_epochs": 5000,
        "save_epochs": 10,
        "policy_update_batch_size": 4096,
        "K_epochs": 5,
        "policy_name": "example",
        "accuracy_terminate_threshold": 0.98,
        "num_render_debug": 4 * 4,
    }
}
