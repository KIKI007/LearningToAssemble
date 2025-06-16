from learn2assemble import ASSEMBLY_RESOURCE_DIR, set_default
from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
from learn2assemble.training import train
import torch


parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/rulin")
settings = {
    "contact_settings": {
        "shrink_ratio": 0.0,
    },
    "env": {
        "name": "rulin",
        "n_robot": 2,
        "boundary_part_ids": [0],
        "sim_buffer_size": 512,
        "num_rollouts": 1024,
        "verbose": False,
    },
    "rbe": {
        "density": 1E4,
        "mu": 0.55,
        "velocity_tol": 1e-2,
        "verbose": False,
    },
    "admm": {
        "Ccp": 1E6,
        "evaluate_it": 100,
        "max_iter": 1000,
        "float_type": torch.float32,
    },
    "search": {
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
        "num_step_anneal": 500,
        "lr_actor": 2e-3,
        "betas_actor": [0.95, 0.999],
    },
    "training": {
        "max_train_epochs": 50000,
        "save_delta_accuracy": 0.01,
        "print_epochs": 1,
        "policy_update_batch_size": 2048,
        "K_epochs": 5
    }
}

contacts = compute_assembly_contacts(parts, settings)
train(parts, contacts, settings)
