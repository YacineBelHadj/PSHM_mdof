
from dynaconf import Dynaconf
from pathlib import Path
from typing import List
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['configuration/settings.toml', 
                    'configuration/.secrets.toml', 
                    'configuration/population.toml'],
    root_path = ROOT_PATH
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
