import os
import sys
# Add top-level package to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from helpers import utils

foo = {'test': 2, 'foo':3}
utils.save_json(foo, 'test/foo')

result_folders = ['test']
print(result_folders)
utils.save_txt(result_folders, 'folders_to_copy.txt')