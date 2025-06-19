from learn2assemble.curriculum import forward_curriculum
def test_curriculum():
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-1")
    contacts = compute_assembly_contacts(parts, default_settings)
    succeed, solution, curriculum = forward_curriculum(parts, contacts, default_settings)
    assert succeed == True