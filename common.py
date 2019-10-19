# --------------------------------------------------------
# Pytorch W2VV++
# Written by Xirong Li & Chaoxi Xu
# --------------------------------------------------------

import os
import logging
#import importlib
from importlib import import_module

ROOT_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')
MIN_WORD_COUNT = 5

TEXT_ENCODINGS = ['bow', 'bow_nsw', 'gru']
DEFAULT_TEXT_ENCODING = 'bow'
DEFAULT_LANG = 'en'

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)


def load_config(config_path):
    module = import_module(config_path)
    return module.config()
