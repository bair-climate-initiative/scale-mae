import os
import threading
from multiprocessing.pool import ThreadPool as Pool
# from threading import Thread

import numpy as np
import scandir
# from tqdm.cli import tqdm

lock = threading.Lock()
from PIL import Image  # NOQA

Image.MAX_IMAGE_PIXELS = 1000000000


def f(item):
    path, _, files = item
    for file in files:
        if ".tif" in file:
            fpath = os.path.join(path, file)
            try:
                img = Image.open(fpath)
                arr = np.array(img)
                arr.shape
            except Exception:
                ...
                # acquire the lock
                lock.acquire()
                # open file for appending
                with open("badfile", "a") as file:
                    # write text to data
                    file.write(fpath + "\n")
                # release the lock
                lock.release()
                print(fpath)


def main():
    directory = "data/naip"
    res = scandir.walk(directory)
    with open("badfile", "w") as file:
        # write text to data
        file.write(">>>>>SOF>>>>>>\n")
    pool = Pool(100)
    pool.map(f, res)


if __name__ == '__main__':
    main()
