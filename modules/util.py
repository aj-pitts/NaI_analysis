import configparser
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def clean_ini_file(input_file, overwrite=False):
    if overwrite:
        output_file = input_file
    else:
        fname, fext = os.path.splitext(input_file)
        output_file = f"{fname}_cleaned{fext}"

    print(f"Reading {input_file}")
    with open(input_file, 'r') as file:
        lines = file.readlines()

    print(f"Writing configuration file to {output_file}...")
    with open(output_file, 'w') as file:
        section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                section = line
                file.write(line + '\n')
            elif '=' in line:
                file.write(line + '\n')
            else:
                if section:
                    file.write(f"{line} = value\n")

    print("Done.")

def check_filepath(filepath,mkdir=True):
    if not os.path.exists(filepath):
        if mkdir:
            print(f"Creating filepath: {filepath}")
            os.mkdir(filepath)
        else:
            raise ValueError(f"'{filepath}' does not exist")

def create_map_plot(image, savepath=None, label=None,deviations=1):
    vmin = int(np.median(image) - deviations * np.std(image))
    vmax = int(np.median(image) + deviations * np.std(image))
    
    w = (image < vmin) | (image > vmax)
    image[w] = np.nan
    
    plt.imshow(image,vmin=vmin,vmax=vmax,origin='lower')
    plt.xlabel("Spaxel")
    plt.ylabel("Spaxel")
    plt.colorbar(label, fraction=0.0465, pad=0.01)
    
    if savepath is not None:
        plt.savefig(savepath,bbox_inches='tight',dpi=200)
        logging.info(f"Saving figure to {savepath}")
        plt.close()
    else:
        plt.show()