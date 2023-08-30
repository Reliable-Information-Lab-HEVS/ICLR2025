# This file is used to automatically extract the results we generated inside the docker container.
# We proceed in this way instead of using a docker volume or mount, because this way the docker instance has
# absolutely no way of hurting the `results` folder since it only sees a copy of it. 
# Modifying the file permissions so that the docker container can only create new files, without modifying 
# or deleting the existing ones seem very hard and cumbersome, if not impossible.

import os
import sys
# Add top-level package to the path (only way to import custom module in a script that is not in the root folder)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from helpers import utils

if __name__ == '__main__':

    # change working directory for relative imports
    os.chdir(utils.ROOT_FOLDER)

    # copy the file indicating which files we will need to copy
    os.system('docker cp my_container:/LLMs/folders_to_copy.txt folders_to_copy.txt')

    folders_to_copy = utils.load_txt('folders_to_copy.txt')

    for folder in folders_to_copy:
        # copy the folder to the host machine (folder should be the relative path, relative to utils.ROOT_FOLDER)
        os.system(f'docker cp my_container:/LLMs/{folder} {folder}')

    # Delete the map of the folders we copied
    os.remove('folders_to_copy.txt')

