from glob import glob
import argparse
import subprocess
import os
from modules import util


def get_args():    
    parser = argparse.ArgumentParser(description="A script to create an equivalent width map of ISM Na I for beta-corrected DAP outputs.")
    
    parser.add_argument('galname',type=str,help='Input galaxy name.')
    parser.add_argument('bin_method',type=str,help='Input DAP patial binning method.')
    parser.add_argument('--redshift',type=str,help='Input galaxy redshift guess.',default=None)
    
    return parser.parse_args()

def main(args):
    repodir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(repodir,'data')
    local_outputs = os.path.join(datadir, 'local_outputs')

    scripts = glob('maps/*.py')
    scripts.remove('maps/__init__.py')

    for script in scripts:
        try:
            subprocess.run(['python', script, args.galname, args.bin_method], check=True)
            print("########################################################################")
            print("########################################################################")
            print(f"##### {script} finished successfully #####")
            print("########################################################################")
            print("########################################################################")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")


    logging.info("Done.")

if __name__ == "__main__":
    args = get_args()
    main(args)