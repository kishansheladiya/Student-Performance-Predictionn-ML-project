"""
Download and extract the UCI Student Performance dataset.
"""
import os
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")


def download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "student.zip")
    if not os.path.exists(zip_path):
        print("Downloading student dataset...")
        urllib.request.urlretrieve(URL, zip_path)
    else:
        print("Zip already downloaded")

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(RAW_DIR)
    print("Extracted files to", RAW_DIR)


if __name__ == "__main__":
    download_and_extract()
