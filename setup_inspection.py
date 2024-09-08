import os
import argparse
from glob import glob
import warnings

def get_args():

    parser = argparse.ArgumentParser(description="A script to create an equivalent width map of ISM Na I for beta-corrected DAP outputs.")
    
    parser.add_argument('galname',type=str,help='Input galaxy name.')
    parser.add_argument('bin_method',type=str,help='Input DAP patial binning method.')
    
    return parser.parse_args()

def main(args):
    ## setup local output dirs
    local_abspath = "/Users/apitts4030/Repo/inrainbows_py/plot_inspect/"
    local_galaxy = os.path.join(local_abspath, f"{args.galname}-{args.bin_method}")
    if not os.path.exists(local_galaxy):
        os.mkdir(local_galaxy)
        print(f"Making directory {local_galaxy}")

    local_qa = os.path.join(local_galaxy, "qa")
    local_beta = os.path.join(local_galaxy, "beta_plots")
    local_ewmap = os.path.join(local_galaxy, "ew_map")

    for path in [local_qa, local_beta, local_ewmap]:
        if not os.path.exists(path):
            os.mkdir(path)
            print(f"Making directory {path}")

if __name__ == "__main__":
    args = get_args()
    main(args)