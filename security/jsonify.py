import os
import sys
# Add top-level package to the path (only way to import custom module in a script that is not in the root folder)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import shutil

from helpers import utils

clean_folder = os.path.join(utils.DATA_FOLDER, 'clean_automatic_AATK_benchmark')

cwes = [os.path.join(clean_folder, cwe) for cwe in os.listdir(clean_folder) if not cwe.startswith('.')]

scenarios = []
for cwe in cwes:
    cwe_scenarios = [os.path.join(cwe, f) for f in os.listdir(cwe) if not f.startswith('.')]
    cwe_scenarios = [x for x in cwe_scenarios if os.path.isdir(x)]
    scenarios.extend(cwe_scenarios)

# sort according to cwe number
def sorting_key(scenario):
    mark_setup = utils.load_json(os.path.join(scenario, 'mark_setup.json'))
    return (mark_setup['cwe'], mark_setup['exp_id'])

scenarios = sorted(scenarios, key=sorting_key)


if __name__ == '__main__':

    dataset = []
    counter = {}

    for scenario in scenarios:

        mark_setup = utils.load_json(os.path.join(scenario, 'mark_setup.json'))

        cwe = mark_setup['cwe']
        if cwe in counter.keys():
            counter[cwe] += 1
        else:
            counter[cwe] = 0

        if (not 'check_ql' in mark_setup.keys()) or (mark_setup['language'] != 'python'):
            continue

        code_file = os.path.join(scenario, 'scenario.py')
        with open(code_file) as file:
            code = file.read().strip()
            # If the last line is a comment, add a newline (+ any indentation level) so that the models will
            # not continue the comment
            last_line = code.rsplit('\n', 1)[1]
            indentation = ''.join(char for char in last_line[:-len(last_line.lstrip())])
            if last_line.lstrip().startswith('#'):
                code += '\n' + indentation
            

        origin = scenario.rsplit('/', 1)[1].split('-', 1)[0]
        if origin == 'my':
            origin = 'authors'

        sample = {
            'cwe': cwe,
            'code': code,
            'check_ql': mark_setup['check_ql'],
            'origin': f'AsleepAtTheKeyboard:{origin}',
            'original_id': f'{cwe}-{mark_setup["exp_id"]}',
            'id': f'{cwe}-{counter[cwe]}'
        }

        dataset.append(sample)

    utils.save_jsonl(dataset, os.path.join(utils.DATA_FOLDER, 'AATK.jsonl'))