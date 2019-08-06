import os
import sys
import logging


logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)


def checkToSkip(filename, overwrite):
    if os.path.exists(filename):
        if overwrite:
            logging.info('%s exists. overwrite', filename)
            return 0
        else:
            logging.info('%s exists. quit', filename)
            return 1
    return 0


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def makedirsforfile(filename):
    makedirs(os.path.split(filename)[0])