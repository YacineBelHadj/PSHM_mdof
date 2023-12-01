In this project we use classification model for anomaly detection, the model is training with the task of discriminating between different system and then the learned embedding is used for anomaly detection using GMM + energy function

## Installation
Note: Everything needs to be done as admin (on Windows).
1) Go to the main directory and run: pip install -e .
2) edit paths in ./configuration/settings.toml
3) in the notebooks: before "from config import settings" add "import sys; sys.path.append(PATH#TO#PROJECT)"
4) add "secrets.toml" file to "./configuration/"

## Running the Code
1) Run ./psm/data/generate_dataset.py
2) Run ./psm/data/process_dataset.py
3) Run any of the notebooks in ./notebooks/experimentation/
