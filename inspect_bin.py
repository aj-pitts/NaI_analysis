import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="A script to inspect the Na D window in a given bin.")
    
    parser.add_argument('galname',type=str,help='Input galaxy name.')
    parser.add_argument('bin_method',type=str,help='Input DAP patial binning method.')

    return parser.parse_args()

def main(args):
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(scriptdir,"bin-inspect")
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    
    


if __name__ == "__main__":
    args = get_args()
    main(args)