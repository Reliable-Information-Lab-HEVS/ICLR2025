import requests
import zipfile
import os
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
    results = os.path.join(DESTINATION, 'results')
    for root, _, files in os.walk(results):
        for file in files:
            if file.startswith('._'):
                os.remove(os.path.join(root, file))



if __name__ == '__main__':
    download()
    clean_up()