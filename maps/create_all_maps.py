from glob import glob
import argparse
import subprocess

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_args():    
    parser = argparse.ArgumentParser(description="A script to create an equivalent width map of ISM Na I for beta-corrected DAP outputs.")
    
    parser.add_argument('galname',type=str,help='Input galaxy name.')
    parser.add_argument('bin_method',type=str,help='Input DAP patial binning method.')
    parser.add_argument('--redshift',type=str,help='Input galaxy redshift guess.',default=None)
    
    return parser.parse_args()

def main(args):
    scripts = glob('*.py')
    scripts.remove('create_all_maps.py')
    scripts.remove('__init__.py')

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