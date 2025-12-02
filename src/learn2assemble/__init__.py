import os
import torch
import numpy as np

RESOURCE_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../data")
ASSEMBLY_RESOURCE_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../data/assembly")


def get_gripper_spheres(open_widths: np.ndarray = None):
    import yaml
    if open_widths is None:
        open_widths = np.array([0.04])
    with open(RESOURCE_DIR + "/gripper/spheres.yml", "r") as stream:
        yml_spheres = yaml.safe_load(stream)
    spheres_list = []
    for name in ["hand", "leftfinger", "rightfinger"]:
        yml_sphere_data = yml_spheres["collision_spheres"][name]
        spheres_list.extend(yml_sphere_data)
        if name == "hand":
            hand_indices = np.arange(0, len(spheres_list))
        if name == "leftfinger":
            left_indices = np.arange(hand_indices[-1] + 1, len(spheres_list))
        if name == "rightfinger":
            right_indices = np.arange(left_indices[-1] + 1, len(spheres_list))
    sphere_centers = np.zeros(shape=(open_widths.shape[0], len(spheres_list), 3), dtype=np.float32)
    sphere_radius = np.zeros(shape=(open_widths.shape[0], len(spheres_list)), dtype=np.float32)
    for id, data in enumerate(spheres_list):
        sphere_centers[:, id, :] = data['center']
        sphere_centers[:, id, 2] += -0.1034
        sphere_radius[:, id] = data['radius']
    sphere_centers[:, left_indices, 1] -= open_widths[:, None]
    sphere_centers[:, right_indices, 1] += open_widths[:, None]
    sphere_radius[:, hand_indices] += 0.008
    return sphere_centers, sphere_radius


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
        "contact_shrink_ratio": 0.1,
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
        "mu": 0.2,
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
        "lr_actor": 1e-5,
        "betas_actor": [0.95, 0.999],

        "per_alpha": 0.8,
        "per_beta": 0.1,
        "per_num_anneal": 500,
    },
    "training": {
        "max_train_epochs": 5000,
        "save_epochs": 5,
        "print_epochs": 1,
        "policy_update_batch_size": 256,
        "K_epochs": 5,
        "policy_name": "example",
        "num_render_debug": 4 * 4,
        "terminate_nondeterminstic_accuracy": 0.9,
        "terminate_determinstic_accuracy": 0.98,
        "terminate_complete_assembly_accuracy": 0.95,
        "full_assembly_ratio": 0.05,
    },
    "grasp":
        {
            "n_sample": 1000,
            "gripper_size": [0.02, 0.02, 0.08],
            "check_ground": True,
            "scale": 0.2,
        },
    "insertion":
        {
            "n_surface_sample": 1000,
            "drt_length": 1,
            "max_dist": 1,
            "collision_eps": 0.01,
            "type": "orthogonal",
            "n_dt_sample": 10,
        }
}
