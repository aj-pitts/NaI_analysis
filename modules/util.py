import configparser
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import warnings
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def clean_ini_file(input_file, overwrite=False):
    """
    Function to clean an .ini configuration file line-by-line if 
    `configparser` returns an error while parsing the file.

    Parameters
    ----------
    input_file : str
        The string containing the filename or filepath to the .ini file
        which must be cleaned.
    overwrite : bool, optional
        Flag to determine whether or not the input file will be
        overwritten. If False, a few file will be created with the same
        filename + "_cleaned" included before the .ini extension.
    """
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
    """
    Checks whether or not a filepath exists using the `os` package.
    If the filepath does not exist, it will be made with mkdir=True

    Parameters
    ----------
    filepath : str
        The filepath to be checked.
    mkdir : bool, optional
        Flag to determine whether or not to make a directory if the
        path does not exist.

    Raises
    ------
    ValueError
        If not `mkdir` and not `os.path.exists(filepath)`.
    ValueError
        If mkdir is True, but the input filepath contains a "."
        indicating a file extension.
    """
    if not os.path.exists(filepath):
        if mkdir:
            if '.' in filepath:
                raise ValueError(f"Filepath cannot contain extensions. {filepath}")
            print(f"Creating filepath: {filepath}")
            os.mkdir(filepath)
        else:
            raise ValueError(f"'{filepath}' does not exist")

def create_map_plot(image, savepath=None, label=None, deviations=1):
    """
    Creates a 2D "map" image of the input values `image` using `matplotlib.pyplot.imshow`.

    Parameters
    ----------
    image : np.ndarray
        The 2D numpy array to be plotted.
    savepath : NoneType or str, optional
        A string of the absolute filepath for the output file. If 
        savepath is None, the image will attempt to be displayed with
        `matplotlib.pyplot.show`.
    label : NoneType or str, optional
        A string containing the label of the value of image to be
        passed into `matplotlib.pyplot.colorbar` as the kwarg label. 
    deviations : int or float, optional
        The value of the number of standard deviations above and below 
        the median to restrict the colorscale of image. vmin and vmax
        are set by 
        int(numpy.median(image) +/- deviations * numpy.std(image)).
    """
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


def get_datapaths(galname, bin_method, logcube=True, maps=True, mcmc=True, config=True, localmaps=True, nc=False):
    """
    Acquires the path(s) to the primary file(s) of a given galaxy by
    request.

    Parameters
    ----------
    galname : str
        The name of the galaxy of which the data will be grabbed.
    bin_method : str
        Spatial binning keyword of the binning method of the desired
        data.
    logcube : bool, optional
        Flag which dicates whether or not to acquire the `LOGCUBE` DAP
        output file. Defaults to BETA-CORR unless `nc` parameter is
        True.
    maps : bool, optional
        Flag which dicates whether or not to acquire the `MAPS` DAP
        output file. Defaults to BETA-CORR unless `nc` parameter is
        True.
    mcmc : bool, optional
        Flag which dicates whether or not to acquire the set of 
        NaImcmcIFU output files.
    config : bool, optional
        Flag which dicates whether or not to acquire the .ini DAP
        configuration file(s).
    localmaps : bool, optional
        Flag which dicates whether or not to acquire the additional
        locally computed maps files.
    nc : bool, optional
        Flag which dictates whether or not to acquire NO-CORR LOGCUBE
        or MAPS files. BETA-CORR files are returned if `nc = False`.
        This parameter has no function if both `logcube` and `maps`
        are set to False

    Returns
    -------
    dict
        A dictionary containing the requested filepath(s) to the
        requested output file(s). The keys of the dictionary are 
        identically named to the input flags. (e.g. the logcube file 
        is stored in the 'logcube' key of the dict). Outputs which are
        likely to contain multiple ("mcmc" and "localmaps") are 
        returned as a list instance.

    Raises
    ------
    ValueError
        When every optional flag is set to false.

    ValueError
        When a file flagged as True cannot be found.
    """

    flags = [logcube, maps, mcmc, config, localmaps]
    if not all(flags):
        raise ValueError("All file flags are set to False. No dictionary to construct...")
    
    outdict = {}
    galdir = f"{galname}-{bin_method}"
    datadir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    if config:
        musecubedir = os.path.join(datadir, "muse_cubes", galname)
        configfils = glob(os.path.join(musecubedir, "*.ini"))
        if len(configfils)>1:
            #raise ValueError(f"{configdir} has more than one config file:\n {configdirfils}")
            #print(f"{musecubedir} has more than one config file:\n {configfils}")
            warnings.warn(f"{musecubedir} has more than one config file:\n {configfils}")
            #print(f"Defaulting to {configfils[0]}")
        else:
            configfils = configfils[0]
        outdict['config'] = configfils
        
    if logcube or maps:
        dapdir = os.path.join(datadir, 'dap_outputs', galdir)
        dapdirfils = glob(os.path.join(dapdir, "**", "*.fits"), recursive=True)
        logcubefil = None
        mapsfil = None

        if nc:
            corr_str = 'NO-CORR'
        else:
            corr_str = 'BETA-CORR'

        for fil in dapdirfils:
            if 'LOGCUBE' in fil and corr_str in fil:
                logcubefil = fil

            if 'MAPS' in fil and corr_str in fil:
                mapsfil = fil

        if logcube:
            if logcubefil is None:
                raise ValueError(f"Could not find BETA-CORR LOGCUBE file in {dapdir}:\n{dapdirfils}")

            outdict['logcube'] = logcubefil

        if maps:
            if mapsfil is None:
                raise ValueError(f"Could not find BETA-CORR MAPS file in {dapdir}:\n{dapdirfils}")
            outdict['maps'] = mapsfil

    if localmaps:
        localmapsdir = os.path.join(datadir,'maps')
        localmapfils = glob(os.path.join(localmapsdir, "*.fits"))
        if len(localmapfils) == 0:
            raise ValueError(f"No local maps found in {localmapsdir}")
        outdict['localmaps'] = localmapfils
        
    if mcmc:
        mcmcdir = os.path.join(datadir, 'mcmc_outputs', galdir)
        mcmcfils = glob(os.path.join(mcmcdir, "**", "*.fits"), recursive=True)
        if len(mcmcfils) == 0:
            raise ValueError(f"No MCMC files found in {mcmcdir}")
        outdict['mcmc'] = mcmcfils

    return outdict