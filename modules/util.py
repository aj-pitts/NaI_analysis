import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
import warnings
import logging
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from modules import defaults
from matplotlib import gridspec

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
                verbose_print(verbose, f"Creating filepath: {path}")
                os.makedirs(path)
            else:
                if error:
                    raise ValueError(f"'{path}' does not exist")
                else:
                    verbose_warning(verbose, f"'{path}' does not exist")


def check_bin_ID(spatial_ID, binid_map, DAPPIXMASK_list = None, stellar_velocity_map = None, s = 5):
    """
    Check whether a spatial bin should be included in analysis based on DAP bin maps, 
    masking criteria, and stellar velocity properties.

    This function evaluates a given spatial bin (`spatial_ID`) against the DAP's `BINID` output, 
    optional masking maps, and stellar velocity constraints to determine if the bin should 
    be excluded from analysis. The bin is flagged if it meets masking criteria, falls outside 
    an acceptable range of stellar velocities, or is otherwise marked invalid in the data.

    Parameters
    ----------
    spatial_ID : int
        Integer identifier of the spatial bin to be checked, corresponding to a value in the 
        `BINID` map from the DAP.

    binid_map : np.ndarray
        Two-dimensional array representing the `BINID` extension 0 map from the DAP, which defines 
        spatial bin assignments.

    DAPPIXMASK_list : list of np.ndarray, optional
        List of two-dimensional arrays representing pixel-level masks from the DAP 
        (`PIXMASK`). These arrays indicate specific reasons a bin may be invalid.

    stellar_velocity_map : np.ndarray, optional
        Two-dimensional array containing stellar velocity data (`STELLAR_VEL`) from the DAP.

    s : int, optional
        Number of standard deviations from the median stellar velocity allowed for a bin to 
        remain valid. Default is 5.

    Returns
    -------
    int
        A flag indicating the bin's validity:
        
        - **0**: Bin passed all checks and is valid for analysis.  
        - **1**: Bin flagged due to pixel-level masking (`DAPPIXMASK_list`).   
        - **2**: Bin's stellar velocity exceeds `s` standard deviations from the median.  

    Notes
    -----
    - If `stellar_velocity_map` is provided, the bin's stellar velocity is compared against the 
      median ± `s` standard deviations.
    - If masking arrays are provided, the function checks each bin against specified mask criteria.
    """
    w = spatial_ID == binid_map

    if stellar_velocity_map is not None:
        median = np.median(stellar_velocity_map[np.isfinite(stellar_velocity_map)])
        std = np.std(stellar_velocity_map[np.isfinite(stellar_velocity_map)])

        sv = np.median(stellar_velocity_map[w])

        threshold = s * std + median
        if abs(sv) > threshold:
            return 2

    if DAPPIXMASK_list is not None:
        dappix_bits = [4, 5, 6, 7, 8, 10, 30]
        for mask_map in DAPPIXMASK_list:
            mask_list = mask_map[w]
            for bitmask in mask_list:
                if bitmask in dappix_bits:
                    return 1
    return 0

def spec_mask_handler(DAPSPECMASK):
    """
    Process DAP spectral mask values and flag bins based on predefined bitmask criteria.

    This function evaluates each bit in the `DAPSPECMASK` array and flags it based on 
    whether it falls within a specific range. Bits with values less than or equal to 9 
    are flagged as invalid (1), while all other bits are considered valid (0).

    See the `DAP Metadata Model <https://sdss-mangadap.readthedocs.io/en/latest/metadatamodel.html#metadatamodel-dapspecmask>`_.

    Parameters
    ----------
    DAPSPECMASK : array-like
        Array of integer bitmask values from the DAP spectral mask (`SPECMASK`), where each 
        value represents a specific data quality flag.

    Returns
    -------
    np.ndarray
        A binary mask array of the same length as `DAPSPECMASK`:
        - **1**: Indicates the corresponding bit is flagged (invalid for analysis).  
        - **0**: Indicates the corresponding bit is valid for analysis.

    Notes
    -----
    - Bits with values <= 9 are considered flagged based on DAP mask criteria.
    - The returned array can be used for filtering or masking spectral data during analysis.
    """
    mask = np.zeros_like(DAPSPECMASK)
    w = (DAPSPECMASK <=9) & (DAPSPECMASK>=2)

    mask[w] = 1
    return mask

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

def unpack_param(param):
    """
    Extracts the value and error from the input parameter.
    
    Parameters
    ----------
    param : float or tuple
        The parameter can be either a float (value) or a tuple (value, error).
    
    Returns
    -------
    value : float
        The main value of the parameter.
    uncertainty : float or None
        The uncertainty associated with the parameter, if provided. Otherwise, None.
    """

    if isinstance(param, tuple):
        value = param[0]
        uncertainty = param[1]
    else:
        value, uncertainty = param, None

    return value, uncertainty


def inspect_bin(galname, bin_method, bin_list, cube_file, map_file, redshift, local_file = None, show_norm = False, verbose = False, QtAgg = True, save = True):
    plt.style.use(os.path.join(defaults.get_default_path('config'), 'figures.mplstyle'))
    if QtAgg:
        matplotlib.use('QtAgg')
    
    cube = fits.open(cube_file)
    maps = fits.open(map_file)

    spatial_bins = cube['binid'].data[0]
    flux = cube['flux'].data 
    wave = cube['wave'].data 
    stellar_cont = cube['model'].data 
    #ivar = cube['ivar'].data

    stellar_vel = maps['stellar_vel'].data 
     
    c = 2.998e5
    NaD_window = (5875, 5915)
    NaD_rest = [5891.5833, 5897.5581]

    if local_file is not None:
        local_maps = fits.open(local_file)
        try:
            mcmc_cube = local_maps['mcmc_results'].data
            mcmc_16 = local_maps['mcmc_16th_perc'].data 
            mcmc_84 = local_maps['mcmc_84th_perc'].data 
        except:
            mcmc_cube = None
            verbose_warning(verbose, f"Local data does not contain 'MCMC_RESULTS'")
    
    ndim = int(np.ceil(np.sqrt(len(bin_list))))
    ncol = min(ndim, 10)
    nrow = int(np.ceil(len(bin_list) / ncol))

    fig_width = ncol * 3
    fig_height = nrow * 3
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    fontsize = max(15, fig.get_size_inches()[1])

    total_subplots = len(bin_list)
    for i in range(total_subplots):
        row_idx = i // ncol  # Current row index
        col_idx = i % ncol   # Current column index
        # Calculate the number of subplots in the last row
        last_row_cols = total_subplots % ncol
        if last_row_cols == 0 and total_subplots != 0:
            last_row_cols = ncol  # If no remainder, the last row fills all columns


        Bin = bin_list[i]
        w = Bin == spatial_bins
        ny, nx = np.where(w)
        y = ny[0]
        x = nx[0]

        z = (stellar_vel[y, x] * (1 + redshift))/c + redshift
        
        restwave = wave / (1 + z)
        flux_1D = flux[:, y, x]
        model_1D = stellar_cont[:, y, x]
        #ivar_1D = ivar[:, y, x]

        wave_window = (restwave >= NaD_window[0]) & (restwave <= NaD_window[1])
        inds = np.where(wave_window)

        gs_parent = gridspec.GridSpec(nrow, ncol, figure=fig)
        subplot_spec = gs_parent[i]  # Get the SubplotSpec
        gs = subplot_spec.subgridspec(3, 1, height_ratios=[2,1,0], hspace=0)
        #gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=fig.add_subplot(nrow, ncol, i+1), height_ratios=[3,1])

        ax1 = fig.add_subplot(gs[0])

        ax1.plot(restwave[inds], flux_1D[inds] / model_1D[inds], 'k', drawstyle='steps-mid')

        #ax.set_title(f"Bin {Bin}")
        ax1.text(.1,.85, f"Bin {Bin}", transform=ax1.transAxes)

        if mcmc_cube is not None:
            lambda_0 = mcmc_cube[0, y, x]
            lambda_16 = mcmc_16[0, y, x]
            lambda_84 = mcmc_84[0, y, x]

            ax1.fill_between([lambda_0-lambda_16, lambda_0+lambda_84], [-20, -20], [20, 20], color='r',
                            alpha=0.3)
            ax1.vlines([lambda_0], -20, 20, colors = 'black', linestyles = 'dashed', linewidths = .6)

        ax1.vlines(NaD_rest, -20, 20, colors = 'dimgray', linestyles = 'dotted', linewidths = .5)
        
        ax1.set_ylim(.85, 1.15)
        ax1.set_xlim(NaD_window[0], NaD_window[1])
        ax1.set_box_aspect(3/4)




        ax2 = fig.add_subplot(gs[1], sharex = ax1)

        min_flux = round(min(flux_1D[inds].min(), model_1D[inds].min()), 2)
        max_flux = round(max(flux_1D[inds].max(), model_1D[inds].max()), 2)
        med_flux = round(max(np.median(flux_1D[inds]), np.median(model_1D[inds])), 2)

        ax2.plot(restwave[inds], flux_1D[inds] / med_flux, 'dimgray', drawstyle = 'steps-mid', lw=1.4)
        ax2.plot(restwave[inds], model_1D[inds] / med_flux, 'tab:blue', drawstyle = 'steps-mid', lw=1.1)

        
        ax2.set_ylim(1 - 0.3, 1 + 0.2)
        ax2.set_box_aspect(3/8)

        if col_idx >= 1:
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])

        if row_idx < nrow - 2:
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
        if row_idx == nrow - 2 and col_idx < last_row_cols:
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])

    fig.text(0.5, 0.05, r'Wavelength $\left( \mathrm{\AA} \right)$', ha='center', va='center', fontsize=fontsize)
    # fig.text(0.05, 0.5, r'Flux (top: Normalized, Bottom: $\left[ \mathrm{1E-17\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}\ spaxel^{-1}} \right]$)',
    #          rotation='vertical',ha='center',va='center', fontsize=fontsize)
    fig.text(0.05, 0.5, r'Normalized Flux', rotation='vertical',ha='center',va='center', fontsize=fontsize)
    #fig.text(0.06, 0.5, r'top: Normalized', rotation='vertical',ha='center',va='center', fontsize=fontsize)
    #fig.text(0.07, 0.5, r'bottom: $\mathrm{1E-17\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}\ spaxel^{-1}}$', rotation='vertical',ha='center',va='center', fontsize=fontsize)

    fig.subplots_adjust(wspace=0.01, hspace=0.05)
    if QtAgg:
        plt.show(block=False)
    else:
        local_data = defaults.get_data_path('local')
        output_dir = os.path.join(local_data, 'local_outputs', f"{galname}-{bin_method}", "figures")
        if not os.path.exists(output_dir):
            raise ValueError(f"Cannot save figure; Path does not exist: {output_dir}")
        if save:
            plt.savefig(os.path.join(output_dir, 'Bin_Inspect.pdf'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()