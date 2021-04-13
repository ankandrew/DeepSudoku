import urllib.request
import os.path


def download_url(url, save_path):
    """
    Modified from: https://stackoverflow.com/a/9419208
    """
    if not os.path.isfile(save_path):
        with urllib.request.urlopen(url) as dl_file:
            with open(save_path, 'wb') as out_file:
                out_file.write(dl_file.read())
