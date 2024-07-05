import os
from glob import glob


def main():
    # input the galaxy name and binning key
    galname = input("Enter Galaxy Name:\n")

    ask = True
    while ask:
        binflag = input("Binkey SQUARE0.6? [Y/N]\n")

        if binflag.lower() == 'y':
            binkey = "SQUARE0.6"
            ask = False
        elif binflag.lower() == 'n':
            binkey = input("Enter binkey:\n")
            ask = False
        else:
            print('Invalid response.')

    # define remote directory of the data
    remote_dir = f"{galname}-{binkey}/"

    # inrainbows paths to the mcmc and cube outputs
    mcmc_path = os.path.join("NaImcmcIFU/muse/NaI_MCMC_output",remote_dir)
    cube_path = os.path.join("mangadap_muse/outputs",remote_dir)

    # set up local paths
    local_path = os.path.dirname(os.path.abspath(__file__))
    local_dir = os.path.join(local_path,f"data/{galname}")
    local_dir_cube = os.path.join(local_dir,"cube/")
    local_dir_mcmc = os.path.join(local_dir,"mcmc/")

    # if directory for the galaxy does not exist, make it
    if not os.path.exists(local_dir):
        print(f"Making directory {local_dir}")
        os.mkdir(local_dir)

    if not os.path.exists(local_dir_cube):    
        print(f"Making directory {local_dir_cube}")
        os.mkdir(local_dir_cube)

    if not os.path.exists(local_dir_mcmc):    
        print(f"Making directory {local_dir_mcmc}")
        os.mkdir(local_dir_mcmc)

    for dir in [local_dir_cube, local_dir_mcmc]:
        if len(glob(f"{dir}/*")) > 0:
            print(f"WARNING: '{dir}' is not an empty directory\n")

    print(f"scp -r apitts@inrainbows:{cube_path} {local_dir_cube}")

    print("\n")

    print(print(f"scp -r apitts@inrainbows:{mcmc_path} {local_dir_mcmc}"))

if __name__ == "__main__":
    main()