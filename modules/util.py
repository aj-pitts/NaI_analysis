import configparser
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import warnings
import logging
import inspect
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
import defaults

def clean_ini_file(input_file, overwrite=False):
    """
    Function to clean an .ini configuration file line-by-line if `configparser` returns an error
    while parsing the file.

    Parameters
    ----------
    input_file : str
        The string containing the filename or filepath to the .ini file which must be cleaned.

    overwrite : bool, optional
        Flag to determine whether or not the input file will be overwritten. If False, a few file
        will be created with the same filename + "_cleaned" included before the .ini extension.
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

def check_filepath(filepath, mkdir=True, verbose=True, error=True):
    """
    Checks whether or not a filepath exists using the `os` package. If the filepath does not exist,
    it will be made with mkdir=True

    Parameters
    ----------
    filepath : str or list
        Either a string of a single filepath or list of strings of filepaths to be checked.

    mkdir : bool, optional
        Flag to determine whether or not to make a directory if the path does not exist.

    verbose : bool, optional
        ...
    error : bool, optional
        ...

    Raises
    ------
    ValueError
        If not `mkdir` and not `os.path.exists(filepath)`.
    ValueError
        If mkdir is True, but the input filepath contains a "." indicating a file extension.
    """
    if isinstance(filepath, str):
        filepath = [filepath]
    
    for path in filepath:
        if not os.path.exists(path):
            if mkdir:
                if '.' in path:
                    raise ValueError(f"Filepath cannot contain extensions. {filepath}")
                verbose_print(verbose, f"Creating filepath: {path}")
                os.mkdir(path)
            else:
                if error:
                    raise ValueError(f"'{path}' does not exist")
                else:
                    verbose_warning(verbose, f"'{path}' does not exist")


def plot_local_maps(mapsfil, verbose = True):
    logger = verbose_logger(verbose=verbose)
    hdul = fits.open(mapsfil)
    mapkeys = [field for field in hdul.keys() if 'MAP' in field.split('_')]


def map_plotter(image: np.ndarray, mask: np.ndarray, label: str, units: str, galname: str, 
             bin_method: str, verbose = True, **kwargs):
    """
    Creates the set of 2D "maps" for a given measurement distribution using `matplotlib.pyplot`.

    Using the 2D numpy arrays `image` and `mask` creates map plots with `matplotlib.pyplot.imshow`.
    The default, guaranteed, map plot is data range limited by \\(s = 3\\) standard deviations above and
    below the median of the masked values of `image`. Optionally, `vmin` and/or `vmax` may be input
    where a new plot will be created with the specified limits. Also plots a histogram of the
    distribution of `image` values. All figures are saved as PDFs by default; if another file type
    is specified with `figext`, it is recommended to specify the figure DPI with the `dpi` kwarg.
    If the map of propagated uncertainties is included as `error`, an additional map of the
    uncertainties will be plotted.

    Parameters
    ----------
    image : np.ndarray
        ...

    mask : np.ndarray
        Boolean or integer mask corresponding to `image`.

    label : str
        The string of the label of the value to be passed into `matplotlib.pyplot` label methods 
        and kwargs.

    units : str
        The string of the units of the value to be passed into `matplotlib.pyplot` label methods
        and kwargs. Recommended to use LaTeX math mode format.

    verbose : bool, optional
        ...

    **kwargs : keyword arguments, optional
        Optional keyword arguments representing additional data. The following keyword arguments 
        are accepted and recommended. Any argument which can be passed into 
        `matplotlib.pyplot.imshow` is also accepted:

        | Keyword     | Type       | Description                                                  |
        |-------------|------------|--------------------------------------------------------------|
        | std         | float      | The number of standard deviations above and below the        |
        |             |            | median to restrict the colorscale of the "std plot". If not  |
        |             |            | specified, the default is 3.                                 |
        | figpath     | str        | The absolute filepath to the figures directory.              |
        |             |            | figures directory.                                           |
        | figname     | str        | The leading name of each figure file to be written.          |
        | figext      | str        | The filetype extension of the figure file.                   |
        | error       | np.ndarray | Map of propagated uncertainties corresponding to `image`.    |
        
    """
    ### init verbose logger
    logger = verbose_logger(verbose)

    plt.style.use(defaults.matplotlib_rc())
    ### unpack and handle defaults for all kwargs

    # names and paths
    figpath = kwargs.get('figpath', defaults.get_default_path('figures'))
    figext = kwargs.get('figext', 'pdf')
    figname = kwargs.get('figname', f'{galname}-{bin_method}-{label}')

    histfigname = os.path.join(figpath, f"{figname}-histogram.{figext}")
    stdfigname = os.path.join(figpath, f"{figname}-map.{figext}")
    
    # values and required maptlotlib kwargs
    s = kwargs.get('std', 3)
    cmap = kwargs.get('cmap', 'rainbow')
    dpi = kwargs.get('dpi', 200)
    errormap = kwargs.get('error', None)

    ### handle required arguments
    mask = mask.astype(bool)
    masked_data = image[~mask]

    median = np.median(image[~mask])
    standard_deviation = np.std(image[~mask])

    value_string = f"{label} {units}"


    #### make the plots
    ## histogram

    #flatten data
    flatdata = np.copy(masked_data).flatten()

    #universal number of bins calculation
    bin_width = 3.5 * np.std(flatdata) / (flatdata.size ** (1/3))
    nbins = (max(flatdata) - min(flatdata)) / bin_width

    plt.hist(flatdata,bins=int(nbins),color='k')
    plt.xlabel(label)
    plt.ylabel(r"$N_{\mathrm{bins}}$")
    plt.xlim(median - 7 * standard_deviation, median + 7 * standard_deviation)

    plt.savefig(histfigname, bbox_inches='tight')
    plt.close()
    logger.info(f"{galname} {label} histogram saved to {histfigname}")


    ## STD map
    vmin = median - s * standard_deviation
    vmax = median + s * standard_deviation

    plotmap = np.copy(image)
    plotmap[mask] = np.nan

    im = plt.imshow(plotmap,cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',
               extent=[32.4, -32.6,-32.4, 32.6])
    plt.xlabel(r'$\alpha$ (arcsec)')
    plt.ylabel(r'$\delta$ (arcsec)')
    ax = plt.gca()
    ax.set_facecolor('lightgray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.01)

    cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
    cbar.set_label(value_string,fontsize=15,labelpad=-45)
    cax.xaxis.set_ticks_position('top')
    

    plt.savefig(stdfigname, bbox_inches='tight',dpi=dpi)
    plt.close()
    logger.info(f"{galname} {label} map plot saved to {stdfigname}")



    ## custom
    vmin = kwargs.get('vmin',None)
    vmax = kwargs.get('vmax',None)

    if vmin or vmax:
        stdcustfigname = os.path.join(figpath, f"{figname}-custom-map.{figext}")
        im = plt.imshow(plotmap,cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',
               extent=[32.4, -32.6,-32.4, 32.6])
        plt.xlabel(r'$\alpha$ (arcsec)')
        plt.ylabel(r'$\delta$ (arcsec)')
        ax = plt.gca()
        ax.set_facecolor('lightgray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.01)

        cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
        cbar.set_label(value_string,fontsize=15,labelpad=-45)
        cax.xaxis.set_ticks_position('top')
        
        plt.savefig(stdcustfigname,bbox_inches='tight',dpi=dpi)
        plt.close()
        logger.info(f"{galname} {label} custom map plot saved to {stdcustfigname}")

    if errormap:
        errormapfigname = os.path.join(figpath, f"{figname}-map-error.{figext}")
        vmin = np.median(errormap[~mask]) - s * np.std(errormap[~mask])
        vmax = np.median(errormap[~mask]) + s * np.std(errormap[~mask])
        error_string = rf"$\sigma_{{{label}}}$ {units}"

        errormap[mask] = np.nan
        im = plt.imshow(errormap,cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',
               extent=[32.4, -32.6,-32.4, 32.6])
        plt.xlabel(r'$\alpha$ (arcsec)')
        plt.ylabel(r'$\delta$ (arcsec)')
        ax = plt.gca()
        ax.set_facecolor('lightgray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.01)

        cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
        cbar.set_label(error_string,fontsize=15,labelpad=-45)
        cax.xaxis.set_ticks_position('top')
        
        plt.savefig(errormapfigname,bbox_inches='tight',dpi=dpi)
        plt.close()
        logger.info(f"{galname} {label} uncertainty map plot saved to {stdcustfigname}")


def check_bin_ID(spatial_ID, binmaps, stellar_velocity_map = None, stellar_vel_mask = None, s = 5, 
                 return_int = True):
    """
    Takes in a spatial bin ID and checks it agains the `BINID` DAP bin maps to check whether
    a bin should be used for analysis.

    First order checks on whether or not analyze a given bin based on the DAP's `BINID` output. The
    function returns an integer flag based on the results of its checks. Optionally adjust the
    `return_int` parameter to allow the function to return a bool instead. An input bin ID
    `spatial_ID` which should not be used for analysis is "flagged". A bin is flagged if the bin is 
    masked by the stellar model fitting results or emission line model fitting results. 
    Optionally include the stellar velocity map and/or the stellar velocity mask parameters to flag
    a bin where its stellar velocity masked or is `s` number of standard deviations away from the
    median.

    Parameters
    ----------
    spatial_ID : int
        Integer identifier of the spatial bin to be checked; from extension 0 of the `BINID` 
        object output by the DAP.

    binmaps : np.ndarray
        `BINID` datacube object output by the DAP. Should contain all five extensions of Bin 
        identifiers.

    stellar_velocity_map : np.ndarray, optional
        Two dimensional array of line-of-sight stellar velocities; `STELLAR_VEL` from the DAP 
        MAPS file.

    stellar_vel_mask : np.ndarray, optional
        Two dimensional array of integer values for masking the stellar velocities; 
        `STELLAR_VEL_MASK` field from the DAP MAPS file.

    s : int, optional
        The number of standard deviations above and below the median stellar velocity which is 
        accepted by the check.

    return_int : bool, optional
        Determines the return type. If `True`, the function returns an integer value corresponding
        to nature of the flagging. If False, it returns a boolean indicating whether or not the
        input bin is flagged. Default is True.

    Returns
    -------
    int or bool
        If `return_int` is `True`:
            - 1: The bin was not flagged by any input and is good to be used.
            - 0: The bin was flagged by either the stellar model results, the emission line model
            results.
            - -1: The bin was flagged by the data quality mask for the stellar velocity results.
            - -2: The stellar velocity of the bin was not flagged, but exceeds `s` standard 
            deviations beyond the median of the distribution of velocities.

        If `return_int` is `False`:
            - True: If the bin was not flagged by any of the checks and is good to use.
            - False: If the bin was flagged and should not be used in analysis.
        
    """
    stellar_vel_mask = stellar_vel_mask.astype(bool)
    spatial_bins = binmaps[0]
    stellar_model_results = binmaps[1]
    emline_model_results = binmaps[3]

    w = spatial_ID == spatial_bins
    ## if bad stellar model or emline fits, skip the spaxel
    if stellar_model_results[w][0] < 0 or emline_model_results[w][0] < 0:
        if not return_int:
            return False
        return 0

    ## if stellar kinematics are masked, skip the spaxel
    if stellar_vel_mask:
        if not stellar_vel_mask[w][0]:
            if not return_int:
                return False
            return -1

    if stellar_velocity_map:
        sv_bin_mask = np.logical_or(w, stellar_vel_mask)
        if abs(stellar_velocity_map[sv_bin_mask][0]) > 5 * np.std(stellar_velocity_map[stellar_vel_mask]) + np.median(stellar_velocity_map[stellar_vel_mask]):
            if not return_int:
                return False
            return -2
        
    if not return_int:
        return True
    return 1

def verbose_print(verbose, *args, **kwargs):
    """
    Prints messages to the console if `verbose` is set to True.

    This function acts as a conditional print, allowing messages to be printed only when verbose 
    mode is enabled. It passes all positional and keyword arguments directly to the built-in 
    `print` function, making it a flexible tool for logging messages in a controlled way.

    Parameters
    ----------
    verbose : bool
        A flag indicating whether to print messages. If True, messages will be printed; if False, 
        no output is generated.

    *args : tuple
        Positional arguments to be passed to `print`. These will be printed in order as they are 
        received.

    **kwargs : dict
        Keyword arguments to be passed to `print`. Common options include `sep` (separator between 
        positional arguments) and `end` (string to append at the end of the print output).

    Examples
    --------
    >>> verbose = True
    >>> verbose_print(verbose, "Processing started...", sep=" - ")
    Processing started...

    >>> verbose = False
    >>> verbose_print(verbose, "This will not be printed.")
    (No output is produced)

    """
    if verbose:
        print(*args, **kwargs)
    
def verbose_logger(verbose):
    """
    Creates and returns a logger that outputs messages to the console if `verbose` is True.

    This logger is configured to output informational messages to the console only when `verbose` 
    mode is enabled. When `verbose` is False, the logger uses a `NullHandler` to suppress output, 
    making it a flexible tool for managing verbosity in scripts.

    Parameters
    ----------
    verbose : bool
        If True, the logger outputs messages to the console; if False, no messages are printed.

    Returns
    -------
    logger : logging.Logger
        A configured logger instance. When `verbose` is True, this logger will output messages with
        level `INFO` or higher to the console. When `verbose` is False, the logger will suppress 
        all output.

    Examples
    --------
    >>> verbose = True
    >>> logger = get_logger(verbose)
    >>> logger.info("This will print because verbose is True.")

    >>> verbose = False
    >>> logger = get_logger(verbose)
    >>> logger.info("This will not print because verbose is False.")

    Notes
    -----
    This function uses Python's built-in `logging` module to manage log output. The logger instance 
    returned can be used throughout the code to produce consistent logging behavior based on the 
    verbosity setting.
    """
    # Set up a logger
    logger = logging.getLogger("VerboseLogger")
    if verbose:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    else:
        # If not verbose, use a null handler (no output)
        logger.addHandler(logging.NullHandler())
    return logger

import warnings

def verbose_warning(verbose, message):
    """
    Issues a warning message if `verbose` is set to True.

    This function provides a controlled way to emit warnings in the script based on the verbosity 
    setting. If `verbose` is True, the warning will be shown; otherwise, no warning is emitted.

    Parameters
    ----------
    verbose : bool
        Flag indicating whether to emit the warning. If True, the warning will be displayed; if 
        False, no warning is issued.

    message : str
        The warning message to be displayed when `verbose` is True.

    Examples
    --------
    >>> verbose = True
    >>> verbose_warning(verbose, "This is a warning message.")
    UserWarning: This is a warning message.

    >>> verbose = False
    >>> verbose_warning(verbose, "This warning will not be shown.")
    (No output is produced)

    Notes
    -----
    This function uses the `warnings` module to control warning messages.
    The `UserWarning` category is used by default, but this can be modified 
    as needed by altering the `warn` function call.
    """
    if verbose:
        warnings.warn(message, category=UserWarning)


def progress_printer(iter_int, max_iter, print_string, verbose = True):
    """
    Prints the current progress of an iterative process in a single-line format.

    Parameters
    ----------
    iter_int : int
        The current iteration number (0-indexed).
    max_iter : int
        The total number of iterations.
    print_string : str
        The base string to print, typically describing the process.
    verbose : bool, optional
        If True, prints the progress message. If False, does nothing (default is True).

    Notes
    -----
    The function prints a progress message that updates in place on the same line. 
    The progress message includes the `print_string` followed by the current iteration 
    and total iterations in the format `<print_string> <current>/<total>`, followed by 
    a rotating dot pattern (`.` to `...`). The rotating dots are intended to visually 
    indicate activity during each iteration.

    Example
    -------
    >>> for i in range(5):
    ...     progress_printer(i, 5, "Processing step")
    Processing step 5/5...
    """

    dots = '.' * ((iter_int % 3) + 1)

    verbose_print(verbose, f"{print_string} {iter_int+1}/{max_iter}{dots}", end='\r')

def redshift_from_config(config_fil, verbose):
    """
    Extracts the redshift value ('z') from a configuration file.

    Parameters
    ----------
    config_fil : str
        The path to the configuration file (typically a `.ini` file).
    verbose : bool
        If True, prints verbose output for errors and cleaning steps during parsing.

    Returns
    -------
    str
        The redshift value ('z') as a string from the 'default' section of the configuration file.

    Notes
    -----
    The function attempts to read the configuration file and parses the redshift 
    value from the 'default' section. If an error occurs while parsing the file, 
    the function will attempt to clean and re-read the file. Any errors during parsing 
    are printed if `verbose` is set to True.

    If the configuration file cannot be parsed correctly, the function will repeatedly 
    clean the file using `modules.util.clean_ini_file` and try to read it again until 
    successful.

    Example
    -------
    >>> z = redshift_from_config("config.ini", verbose=True)
    >>> print(z)
    '0.023'
    """

    config = configparser.ConfigParser()
    parsing = True
    while parsing:
        try:
            config.read(config_fil)
            parsing = False
        except configparser.Error as e:
            verbose_print(verbose, f"Error parsing file: {e}")
            verbose_print(verbose, f"Cleaning {config_fil}")
            clean_ini_file(config_fil, overwrite=True)


    redshift = config['default']['z']

    return redshift