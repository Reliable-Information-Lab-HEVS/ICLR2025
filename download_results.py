import requests
import zipfile
import subprocess
import io

from helpers import utils

URL = 'https://www.dropbox.com/scl/fi/9yz9g7924nxwb8lmuazqw/results.zip?rlkey=h5ftdfyih8ve5awht4padsv0s&st=t0ozrq3i&dl=1'
DESTINATION = utils.ROOT_FOLDER

def download():

    response = requests.get(URL)
    
    if response.status_code == 200:
        print('Files downloaded successfully')
        # BytesIO avoids using a temporary file to store the bytes response
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip:
            zip.extractall(DESTINATION)
    else:
        raise RuntimeError('Failed to download files')


def clean_up():
    # Those annoying files starting with "._" were created when copying the data -> remove them
    clean_command = 'find . -name "._*" -type f -delete'
    p = subprocess.run(clean_command, shell=True)
    if p.returncode != 0:
        raise RuntimeError('Error when cleaning-up.')


if __name__ == '__main__':
    download()
    clean_up()