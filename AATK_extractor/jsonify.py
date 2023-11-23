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
    cwe = mark_setup['cwe']
    cwe_number = int(cwe.replace('CWE-', ''))
    return (cwe_number, int(mark_setup['exp_id']))

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
            code = file.read()
            
        origin = scenario.rsplit('/', 1)[1].split('-', 1)[0]
        if origin == 'my':
            origin = 'authors'

        # Create new folder where we save the custom ql queries for easier access
        folder = os.path.join(utils.DATA_FOLDER, 'custom_ql_queries')
        os.makedirs(folder, exist_ok=True)

        # Check if the query is custom or not
        query_path = mark_setup['check_ql']
        if not query_path.startswith('$CODEQL_HOME'):
            actual_path = query_path.replace('./experiments_dow', clean_folder)
            cwe_path = os.path.dirname(os.path.dirname(actual_path))
            qlpack = os.path.join(cwe_path, 'qlpack.yml')

            # need to create an intermediary directory to add the qlpack.yml files
            filename = actual_path.rsplit('/', 1)[1]
            intermediate_folder = filename.rsplit('.', 1)[0]
            new_folder = os.path.join(folder, intermediate_folder)
            os.makedirs(new_folder, exist_ok=True)

            # copy the needed files
            shutil.copyfile(actual_path, os.path.join(new_folder, filename))
            shutil.copyfile(qlpack, os.path.join(new_folder, 'qlpack.yml'))

            # add the query path as a relative path (containing '.' at the start) to the sample
            query_path = os.path.join('.', os.path.relpath(os.path.join(new_folder, filename), utils.ROOT_FOLDER))


        sample = {
            'cwe': cwe,
            'code': code,
            'stopping': mark_setup['stopping'],
            'check_ql': query_path,
            'language': mark_setup['language'],
            'id': f'{cwe}-{counter[cwe]}',
            'origin': f'AsleepAtTheKeyboard:{origin}',
            'original_id': f'{cwe}-{mark_setup["exp_id"]}',
        }

        dataset.append(sample)

    utils.save_jsonl(dataset, os.path.join(utils.DATA_FOLDER, 'AATK.jsonl'))
