import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import table
from glob import glob
import os
import argparse


def make_vmap():

    return

    
def get_args():
    parser = argparse.ArgumentParser(description="A script to create an Na I velocity map from the MCMC results of a beta-corrected galaxy.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument()

    return parser.parse_args()


def main(args):
    script_dir = os.path.abspath(__file__)
    

    output_dir
    


if __name__ == "__main__":

    main()