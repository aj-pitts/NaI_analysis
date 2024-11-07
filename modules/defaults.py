import os
from pathlib import Path
import time

def get_default_path(subdir=None, ensure_exists=True):
    """
    Get a default path for a specific subdirectory within the repository.
    If no subdirectory is specified, returns the path to the repository root.

    Parameters
    ----------
    subdir : str or None
        Subdirectory within the repository (e.g., 'data', 'config'). If None,
        returns the path to the repository root directory.
    ensure_exists : bool
        If True, creates the directory if it does not exist (only if subdir is specified).

    Returns
    -------
    pathlib.Path
        Path to the specified subdirectory, or the repository root if subdir is None.
    """
    # Define the root path as the parent directory of this script
    root_path = Path(__file__).parent.parent

    # Return root path if no subdirectory is specified
    if subdir is None:
        return root_path

    # Otherwise, return the specified subdirectory path
    path = os.path.join(root_path, subdir)
    if ensure_exists:
        if not os.path.exists(path):
            raise ValueError(f"{subdir} is not a directory within {root_path}")
    return path


def timer(func):
    """
    Decorator to time a function's execution.

    Parameters
    ----------
    func : function
        The function to be timed.

    Returns
    -------
    function
        Wrapped function with timing output.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper


def matplotlib_rc():
    """
    Get the path to the custom Matplotlib style configuration file.

    This function returns the path to the `figures.mplstyle` file located in
    the 'config' subdirectory of the repository. The file is intended to store
    custom Matplotlib style settings for consistent plot styling across scripts.

    Returns
    -------
    str
        Absolute path to the `figures.mplstyle` configuration file.
    """
    return os.path.join(get_default_path('config'), 'figures.mplstyle')

def local_quality_flag():
    """
    Generate a dictionary of quality flags and their descriptions for data bins.

    This function returns a dictionary where each key represents a flag value
    (integer) and each corresponding value is a string description of the 
    flag's meaning. The flags are used to assess the quality of data bins 
    in stellar and emission line models, indicating whether a bin should 
    be included in further analysis or discarded based on certain criteria.

    Returns
    -------
    dict
        A dictionary of integer keys and string descriptions of integer quality flag meaning:

    Examples
    --------
    >>> flags = local_quality_flag()
    >>> print(flags[1])
    "The bin was not flagged by any input and is good to be used."
    
    >>> print(flags[-3])
    "The computed value is NaN"
    """
    
    flags = {1:'The bin was not flagged by any input and is good to be used.',
             0:'The bin was flagged by either the stellar model results, the emission line model results.',
             -1:'The bin was flagged by the data quality mask for the stellar velocity results.',
             -2:'The stellar velocity of the bin was not flagged, but exceeds the number of standard deviations beyond the median of the distribution of velocities allowed during the computation.',
             -3:'The computed value is NaN',
             -4:'The computed uncertainty is NaN'
             }
    return flags