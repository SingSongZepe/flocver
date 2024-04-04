import os

from utils.log import *

# make dir if not exists or lost
def make_dir(path: str) -> bool:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        ln(f'create {path} successfully')
    return True

