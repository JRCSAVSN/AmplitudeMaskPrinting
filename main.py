import torch
import numpy as np
from PIL import Image

import argparse
import matplotlib.pyplot as plt

from modules import gen_patterns
from modules.gen_patterns import *
from modules.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to configuration file')
args = parser.parse_args()

if __name__ == '__main__':
    CFG = load_config(args.config)

    gen_fn = getattr(gen_patterns, 'gen_' + CFG['pattern'])

    params = CFG.get('params', {})
    mask = gen_fn(**params)

    # Slice mask into N files of max height CFG['slicing']['max_height'] pixels
    max_height = CFG.get('slicing', {'max_height': mask.shape[0]}).get('max_height', mask.shape[0])
    for i in range((mask.shape[0]//CFG['slicing']['max_height'])):
        Image.fromarray(mask[i*max_height:(i+1)*max_height]).save(CFG['save_path'] + f'{i}.bmp')

