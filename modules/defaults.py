import os
from pathlib import Path
import time
import json

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

def get_data_path(subdir=None):
    """
    Retrieve the data path from the configuration file, optionally for a specific subdirectory.

    This function reads the `config.json` file from the default configuration directory,
    extracts the `data_path` value, and returns it. An optional `subdir` argument can specify
    a particular subdirectory mode.

    Parameters
    ----------
    subdir : str or None, optional
        Specifies the subdirectory mode. Must be one of {None, 'pipeline', 'local'}.
        If None (default), no subdirectory mode is applied.

    Returns
    -------
    str
        The absolute path to the data directory as specified in the `config.json` file.

    Raises
    ------
    ValueError
        If `subdir` is not one of {None, 'pipeline', 'local'}.
    FileNotFoundError
        If the `config.json` file does not exist in the expected location.
    KeyError
        If the `data_path` key is missing in the `config.json` file.
    json.JSONDecodeError
        If the `config.json` file contains invalid JSON.
    """
    valid_subdirs = {None, 'pipeline', 'local'}
    if subdir not in valid_subdirs:
        raise ValueError(f"{subdir} is not a valid subdirectory. Must be one of {valid_subdirs}")
    
    config_path = get_default_path('config')
    config_filepath = os.path.join(config_path, 'config.json')
    with open(config_filepath) as config_file:
        config = json.load(config_file)
    
    data_path = config['data_path']

    if subdir is not None:
        data_path = os.path.join(data_path, subdir)

    if not os.path.exists(data_path):
        raise ValueError(f"Filepath does not exist: {data_path}")
    
    return data_path


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
    """
    
    flags = {0:'No mask',
             1:'The bin contained spaxels masked by the MANGA_DAPSPECMASK.',
             2:'The bin contained spaxels masked by the MANGA_DAPPIXMASK.',
             3:'The bin STELLAR_VEL exceeds the user-set threshold.',
             4:'The computed value is NaN.',
             5:'The computed uncertainty is NaN.',
             6:'Spaxel not included in DAP.',
             7:'Bin S/N is below threshold.',
             8:'Measurement uncertainty is beyond user-set threshold.'
             }
    return flags

def analysis_plans():
    """
    Returns a dictionary of the current default analysis plan methods.

    Returns
    -------
    str
        'MILESHC-MASTARSSP-NOISM'

    """
    return 'MILESHC-MASTARSSP-NOISM'