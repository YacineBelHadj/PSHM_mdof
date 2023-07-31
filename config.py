
from dynaconf import Dynaconf
from pathlib import Path
from typing import List

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['configuration/settings.toml', 
                    'configuration/.secrets.toml', 
                    'configuration/population.toml'],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
