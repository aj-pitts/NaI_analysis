import os
from glob import glob
import argparse


def get_args():
    """
    Command line input arguments.

    Returns:
            :class:`argparse.Namespace`: Converts argument
            strings to objects and assigns them as attributes to the class `Namespace`.
    """
    parser = argparse.ArgumentParser(description='File to run the MaNGA DAP on a MUSE cube.')

    parser.add_argument('galname', type=str,help='input galaxy name.')

    parser.add_argument('bin_key', type=str, help='input DAP spatial binning method.')

    return parser.parse_args()



def main(args):
    galname = args.galname
    binkey = args.bin_key

    # define remote directory of the data
    gal_dir = f"{galname}-{binkey}/"
    inrainbows_dir = "/data2/muse/"

    # inrainbows paths to the mcmc and cube outputs
    mcmc_path = os.path.join(inrainbows_dir,"mcmc_outputs",gal_dir)
    cube_path = os.path.join(inrainbows_dir,"dap_outputs",gal_dir)
    ini_path = os.path.join(inrainbows_dir,"muse_cubes",f"{galname}/*.ini")

    # set up local paths
    local_path = os.path.dirname(os.path.abspath(__file__))
    local_dir = os.path.join(local_path,f"data/{galname}")

    if not os.path.exists(local_dir):
        print(f"Making directory {local_dir}")
        os.mkdir(local_dir)

    local_dir_cube = os.path.join(local_dir,"cube/")
    local_dir_mcmc = os.path.join(local_dir,"mcmc/")
    local_dir_ini = os.path.join(local_dir,"config/")

    dir_list = [local_dir_cube, local_dir_mcmc, local_dir_ini]

    # if directory for the galaxy does not exist, make it
    for dir in dir_list:
        if not os.path.exists(dir):
            print(f"Making directory {dir}.\n")
            os.mkdir(dir)

    # warn if a directory is not empty
    for dir in dir_list:
        if len(glob(f"{dir}/*")) > 0:
            print(f"WARNING: '{dir}' is not an empty directory\n")


    print("SCP COMMANDS:\n")
    print(f"scp -r apitts@inrainbows:{cube_path} {local_dir_cube}")
    print(f"scp -r apitts@inrainbows:{mcmc_path} {local_dir_mcmc}")
    print(f"scp -r 'apitts@inrainbows:{ini_path}' {local_dir_ini}")
    print("\n\n")



if __name__ == "__main__":
    args = get_args()
    main(args)