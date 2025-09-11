# module initialisation loads the functions and self set attributes from the setting_loader module, this may be after updating the values from ones defined in yaml files.
# settings/__init__.py
import sys
from .setting_loader import *

# Initialize module reference
_current_module = sys.modules[__name__]


# Expose core configuration objects
__all__ = [
    "settings_from_files",
    "f_settings",
    "rate_constants",
    "region_remapping",
    "get_rate_constant",
]

# Initialize default values
rate_constants = {}
region_remapping = {}


def __init_settings():
    """Initialize settings from files"""
    global rate_constants, region_remapping
    f_settings = settings_from_files

    if OVERRIDE_SETTINGS_WITH_FILESETTINGS:
        # Update module attributes with loaded settings
        rate_constants.update(settings_from_files.rate_regions.content)
        region_remapping.update(settings_from_files.id_regions.content)

        # Mirror attributes at module level
        for k, v in settings_from_files.rate_regions.content.items():
            setattr(_current_module, k, v)


# Automatically initialize when imported
__init_settings()
