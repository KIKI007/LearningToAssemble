import os
RESOURCE_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../data")
ASSEMBLY_RESOURCE_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../data/assembly")

def set_default(settings: dict,
                name: str,
                default: dict):
    sub_settings = settings.get(name, {})
    for item_name, value in default.items():
        if item_name not in sub_settings:
            sub_settings[item_name] = value
    settings[name] = sub_settings
    return sub_settings
