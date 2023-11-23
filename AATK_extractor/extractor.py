import os
import sys
# Add top-level package to the path (only way to import custom module in a script that is not in the root folder)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import shutil

from helpers import utils

original_folder = os.path.join(utils.DATA_FOLDER, 'original_AATK_benchmark')
new_folder = os.path.join(utils.DATA_FOLDER, 'automatic_AATK_benchmark')

# copy all files, then we will delete scenarios that do not interest us
shutil.copytree(original_folder, new_folder)

cwes = [os.path.join(new_folder, cwe) for cwe in os.listdir(new_folder) if not cwe.startswith('.')]

scenarios = []
for cwe in cwes:
    cwe_scenarios = [os.path.join(cwe, f) for f in os.listdir(cwe) if not f.startswith('.')]
    cwe_scenarios = [x for x in cwe_scenarios if os.path.isdir(x)]
    scenarios.extend(cwe_scenarios)


if __name__ == '__main__':

    for scenario in scenarios:

        mark_setup = utils.load_json(os.path.join(scenario, 'mark_setup.json'))

        if (not 'check_ql' in mark_setup.keys()) or (mark_setup['language'] != 'python'):
            shutil.rmtree(scenario)

        # code_file = os.path.join(scenario, 'scenario.py')
        # with open(code_file) as file:
        #     code = file.read()

        # out = {
        #     'cwe': mark_setup['cwe'],
        #     'check_ql': mark_setup['check_ql'],
        #     'code': code,
        #     'origin': 'AsleepAtTheKeyboard',
        # }

    # Delete empty cwe folders
    for cwe in cwes:
        childs = [os.path.join(cwe, f) for f in os.listdir(cwe) if not f.startswith('.')]
        if not any(os.path.isdir(child) for child in childs):
            shutil.rmtree(cwe)