from os.path import abspath, dirname, join


SRC_DIR_PATH = abspath(dirname(__file__))

RAW_DATA_DIR_PATH = abspath(join(SRC_DIR_PATH, "../data/raw"))
INTERIM_DATA_DIR_PATH = abspath(join(SRC_DIR_PATH, "../data/interim"))
PROCESSED_DATA_DIR_PATH = abspath(join(SRC_DIR_PATH, "../data/processed"))

LOG_DIR_PATH = abspath(join(SRC_DIR_PATH, "../logs"))
CKPT_DIR_PATH = abspath(join(SRC_DIR_PATH, "../checkpoints"))
