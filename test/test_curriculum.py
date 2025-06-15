from learn2assemble.curriculum import forward_curriculum
def test_curriculum():
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, set_default
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import torch

    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-1")

    settings = {
        "contact_settings": {
            "shrink_ratio": 0.0,
        },
        "rbe": {
            "density": 1E2,
            "mu": 0.55,
            "velocity_tol": 1e-2,
            "verbose": False,
        },
        "admm": {
            "Ccp": 1E6,
            "evaluate_it": 200,
            "max_iter": 2000,
            "float_type": torch.float32,
        },
        "search": {
            "n_beam": 64
            "verbose": False,
        },
        "env": {
            "n_robot": 2,
            "boundary_part_ids": [0],
        }
    }

    contacts = compute_assembly_contacts(parts, settings)
    succeed, solution, curriculum = forward_curriculum(parts, contacts, settings)
    assert succeed == True