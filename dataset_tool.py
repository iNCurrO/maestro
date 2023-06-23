import gdown
from config import get_config
import os
import zipfile


def main():
    url = 'https://drive.google.com/uc?id=19lGLrBsIZfUfhOaGTv29cbWRGTiX50_z'
    print("Parse configurations")
    config = get_config()
    if not os.path.exists(os.path.join(config.datadir)):
        os.mkdir(os.path.join(config.datadir))
    if not os.path.exists(os.path.join(config.datadir, "figuredatas")):
        gdown.download(url, os.path.join(config.datadir, "downloadedfromGD_FD"), fuzzy=True)
        extract_and_remove("downloadedfromGD_FD", "figuredatas", config)

def extract_and_remove(filename, targetdir, config):
    with zipfile.ZipFile(os.path.join(config.datadir, filename), 'r') as zip_ref:
        print("Start extracting... ")
        zip_ref.extractall(os.path.join(config.datadir, targetdir))
    print("Removing zip file... ")
    os.remove(os.path.join(config.datadir, filename))


if __name__ == "__main__":
    main()
