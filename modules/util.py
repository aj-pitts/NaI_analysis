import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import warnings
import logging
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

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