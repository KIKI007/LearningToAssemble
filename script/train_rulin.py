from learn2assemble import ASSEMBLY_RESOURCE_DIR, set_default
from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
from learn2assemble.training import train, evaluation
from learn2assemble.render import render_batch_simulation
import torch
from multiprocessing import Queue, Process

if __name__ == "__main__":
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/rulin")
    settings = {
        "contact_settings": {
            "shrink_ratio": 0.0,
        },
        "env": {
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
            "n_beam": 128,
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
            "max_train_epochs": 1000,
            "save_delta_accuracy": 0.01,
            "print_epochs": 1,
            "policy_update_batch_size": 4096,
            "K_epochs": 10,
            "output_name": "rulin",
            "num_render_debug": 4 * 4,
            "accuracy_terminate_threshold": 0.98
        }
    }

    contacts = compute_assembly_contacts(parts, settings)
    train(parts, contacts, settings, None)

    # gui
    evaluation_gui = False
    queue = Queue()
    if not evaluation_gui:
        evaluation(parts, contacts, "rulin", 4*4,None)
    else:
        p1 = Process(target=evaluation, args=(parts, contacts, 'rulin', 4*4, queue))
        p2 = Process(target=render_batch_simulation, args=(parts, settings["env"]["boundary_part_ids"], queue))
        p2.start()
        p1.start()
        p1.join()
        p2.join()