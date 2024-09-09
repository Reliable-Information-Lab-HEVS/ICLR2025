import requests
import zipfile
import os
import io
from tqdm import tqdm

from helpers import utils

URL = 'https://www.dropbox.com/scl/fi/9yz9g7924nxwb8lmuazqw/results.zip?rlkey=h5ftdfyih8ve5awht4padsv0s&st=t0ozrq3i&dl=1'
DESTINATION = utils.ROOT_FOLDER


def download():
    response = requests.get(URL, stream=True)
    if response.status_code != 200:
        raise RuntimeError('Failed to download files')
    
    total_byte_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 ** 2
    iterations = total_byte_size // chunk_size + 1
    byte_chunks = []
    for data in tqdm(response.iter_content(chunk_size), total=iterations, desc='Downloading', unit='MiB'):
        byte_chunks.append(data)

    full_byte_response = b''.join(byte_chunks)
    # BytesIO avoids using a temporary file to store the bytes response
    with zipfile.ZipFile(io.BytesIO(full_byte_response)) as zip:
        zip.extractall(DESTINATION)


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