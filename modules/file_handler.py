from astropy.io import fits
import os
from glob import glob
import numpy as np
import util
import time
import sys
import defaults
import warnings
import configparser

# TODO: update the docstring, add verbose printing, add better bc/nc option

# 
# 
# Data acquisition
# 
# 

def init_datapaths(galname, bin_method, verbose=False):
    """
    Acquires the path(s) to the primary file(s) of a given galaxy by request.

    Parameters
    ----------
    galname : str
        The name of the galaxy of which the data will be grabbed.

    bin_method : str
        Spatial binning keyword of the binning method of the desired data.

    verbose : bool, optional
        ...

    Returns
    -------
    dict
        A dictionary containing relevant filepaths for the input galaxy and binning method.

        **Dictionary Structure**:

        • **'CONFIG'** (str): Path to the `.ini` configuration file.

        • **'NO-CORR'** (dict): Dictionary with filepaths for non-corrected data:
        
            - 'LOGCUBE' (str): Path to the non-corrected logcube file.
            - 'MAPS' (str): Path to the non-corrected maps file.

        • **'BETA-CORR'** (dict): Dictionary with filepaths for beta-corrected data:

            - 'LOGCUBE' (str): Path to the beta-corrected logcube file.
            - 'MAPS' (str): Path to the beta-corrected maps file.
            - 'MCMC' (list): List of paths for MCMC results.
            - 'LOCAL' (str): Path to the locally corrected data.

        **Note**: If a file does not exist, its value in the dictionary will be `None`.
    """
    util.verbose_print(verbose, f"Acquiring relevant files for {galname}-{bin_method}")
    ## initialize paths to the data directory and relevant subdirectories
    repopath = defaults.get_default_path()
    datapath = os.path.join(os.path.dirname(repopath))

    galsubdir = f"{galname}-{bin_method}"

    dappath = os.path.join(datapath, "DAP_outputs", galsubdir)
    mcmcpath = os.path.join(datapath, "MCMC_outputs", galsubdir)
    musecubepath = os.path.join(datapath, "MUSE_cubes", galname)
    localdatapath = os.path.join(datapath, "LOCAL_outputs", galsubdir)

    outdict = {'CONFIG':None, 
               'NO-CORR':{'LOGCUBE':None, 'MAPS':None}, 
               'BETA-CORR':{'LOGCUBE':None, 'MAPS':None, 'MCMC':None, 'LOCAL':None}}
    
    ### CONFIG file
    configfils = glob(os.path.join(musecubepath, "*.ini"))
    if len(configfils) == 0:
        util.verbose_warning(verbose,f"No configuration file found in {musecubepath}")

    else:
        if len(configfils)>1:
            warnings.warn(f"{musecubepath} has more than one config file:\n{configfils} \nDefaulting to {configfils[0]}")

        outdict['CONFIG'] = configfils[0]

    ### DAP files
    if not os.path.exists(dappath):
        util.verbose_warning(verbose,f"Filepath does not exist: {dappath}")
    else:
        dapdirfils = glob(os.path.join(dappath, "**", "*.fits"), recursive=True)
        if len(dapdirfils) != 0:
            for fil in dapdirfils:
                if 'LOGCUBE' in fil:
                    if 'NO-CORR' in fil:
                        outdict['NO-CORR']['LOGCUBE'] = fil
                    elif 'BETA-CORR' in fil:
                        outdict['BETA-CORR']['LOGCUBE'] = fil

                if 'MAPS' in fil:
                    if 'NO-CORR' in fil:
                        outdict['NO-CORR']['MAPS'] = fil
                    elif 'BETA-CORR' in fil:
                        outdict['BETA-CORR']['MAPS'] = fil
        else:
            util.verbose_warning(verbose, f"No DAP files found in {dapdirfils}")

    ### MCMC files
    if not os.path.exists(mcmcpath):
        util.verbose_warning(verbose,f"Filepath does not exist: {mcmcpath}")

    else:
        mcmcfils = glob(os.path.join(mcmcpath, "**", "*.fits"), recursive=True)
        if len(mcmcfils) != 0:
            outdict['BETA-CORR']['MCMC'] = mcmcfils
        else:
            util.verbose_warning(verbose, f"No MCMC files found in {mcmcpath}")

    ### LOCAL file
    if not os.path.exists(localdatapath):
        util.verbose_warning(verbose,f"Filepath does not exist: {localdatapath}")
    else:
        localfil = glob(os.path.join(localdatapath, ".fits"))
        if len(localfil) == 0:
            util.verbose_warning(verbose,f"No local data file found in {localdatapath}")
        else:
            outdict["BETA-CORR"]['LOCAL'] = localfil[0]

    return outdict

# 
# 
# Section for handling FITS file writing
# format headers, automatically write 2D maps into FITS Files
# 
# 

def standard_header_dict(galname, HDU_keyword, unit_str, flag):
    """
    Creates a standard header dictionary for FITS file extensions based on the given flag.

    Parameters
    ----------
    galname : str
        The name of the galaxy associated with the data.
    HDU_keyword : str
        The base keyword related to the HDU (Header Data Unit) for the extension.
    unit_str : str
        The string representing the units for the pixel values (e.g., "Jy/beam").
    flag : str
        The type of data for which the header is being generated. Valid values are:
        - "data" : Standard data header for the map.
        - "err"  : Header for the error/uncertainty data.
        - "mask" : Header for the mask data.
        - "qual" : Header for the quality flags data.

    Returns
    -------
    dict
        A dictionary containing standard header entries for the corresponding data type.
        The keys are header keywords, and the values are tuples containing the associated 
        values and descriptions.

    Notes
    -----
    The structure of the returned dictionary depends on the value of the `flag` parameter:

    - **For "data"**: 
      Includes fields like "DESC", "BUNIT", "ERRDATA", "QUALDATA", "FLAGDATA", "EXTNAME", and "AUTHOR".
    
    - **For "err"**:
      Includes fields like "BUNIT", "DATA", "QUALDATA", "FLAGDATA", "EXTNAME", and "AUTHOR".
    
    - **For "mask"**:
      Includes fields like "ERRDATA", "FLAGDATA", "EXTNAME", and "AUTHOR".
    
    - **For "qual"**:
      Includes quality flag descriptions ("FLG_1", "FLG_0", etc.), "ERRDATA", "QUALDATA", "EXTNAME", and "AUTHOR".

    The function assigns appropriate extension names, data descriptions, and associated metadata 
    based on the type of data (`flag`).

    Example
    -------
    >>> header = standard_header_dict("NGC_1234", "H_ALPHA", "Jy/beam", "data")
    >>> print(header["DESC"])
    ("NGC_1234 h alpha map", "")
    """

    if flag == "data":
        header = {
            "DESC":(f"{galname} {HDU_keyword.lower().replace("_"," ")} map",""),
            "BUNIT":(unit_str, "Unit of pixel value"),
            "ERRDATA":(f"{HDU_keyword}_ERR", "Associated uncertainty values extension"),
            "QUALDATA":(f"{HDU_keyword}_MASK", "Associated quality extension"),
            "FLAGDATA":(f"{HDU_keyword}_FLAG", "Associated quality flags extension"),
            "EXTNAME":(HDU_keyword, "extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    
    elif flag == "err":
        header = {
            "BUNIT":(unit_str, "Unit of pixel value"),
            "DATA":(HDU_keyword, "Associated data extension"),
            "QUALDATA":(f"{HDU_keyword}_MASK", "Associated quality extension"),
            "FLAGDATA":(f"{HDU_keyword}_FLAG", "Associated quality flags extension"),
            "EXTNAME":(f"{HDU_keyword}_ERR", "extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    
    elif flag == "mask":
        header = {
            "ERRDATA":(f"{HDU_keyword}_ERR", "Associated uncertainty values extension"),
            "FLAGDATA":(f"{HDU_keyword}_FLAG", "Associated quality extension"),
            "EXTNAME":(f"{HDU_keyword}_MASK", "extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    
    elif flag == "qual":
        quality = defaults.local_quality_flag()
        header = {
            "FLG_1":(quality[1],"Description of quality flag of 1"),
            "FLG_0":(quality[0],"Description of quality flag of 0"),
            "FLG_-1":(quality[-1],"Description of quality flag of -1"),
            "FLG_-2":(quality[-2],"Description of quality flag of -2"),
            "FLG_-3":(quality[-3],"Description of quality flag of -3"),
            "FLG_-4":(quality[-4],"Description of quality flag of -4"),
            "ERRDATA":(f"{HDU_keyword}_ERR", "Associated uncertainty values extension"),
            "QUALDATA":(f"{HDU_keyword}_MASK", "Associated quality extension"),
            "EXTNAME":(f"{HDU_keyword}_FLAG", "extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    
    return header


def standard_map_dict(galname, HDU_keyword, unit_str, mapdict):
    """
    Creates a standardized dictionary mapping for the given galaxy name and HDU keyword.

    Parameters
    ----------
    galname : str
        The name of the galaxy for which the mapping is being generated.
    HDU_keyword : str
        The base keyword to be used for creating HDU-related keys in the dictionary.
    unit_str : str
        The unit string to be used in the header information.
    mapdict : dict
        A dictionary containing the map data with each key corresponding to a different data array.

    Returns
    -------
    dict
        A dictionary containing standardized mappings for the galaxy's data. The dictionary has
        the following structure:
        
        - Keys are based on the provided `HDU_keyword` and include additional suffixes for 
          mask, error, and flag data.
        - Values are tuples containing the data from `mapdict` and the corresponding header dictionary 
          generated using `standard_header_dict`.

    Notes
    -----
    The function constructs a dictionary of the following form:
    
    - `<HDU_keyword>` : tuple of (data, header_dict)
    - `<HDU_keyword>_MASK` : tuple of (mask_data, header_dict)
    - `<HDU_keyword>_ERROR` : tuple of (error_data, header_dict)
    - `<HDU_keyword>_FLAG` : tuple of (flag_data, header_dict)

    The header dictionary is created using the `standard_header_dict` function, which includes the
    galaxy name, HDU keyword, unit string, and a specific flag for each type of data.

    **Important**: The `mapdict` must be ordered correctly, as the order of the values in `mapdict` 
    directly determines the keys of the output dictionary. If the order is incorrect, the corresponding 
    data may not be correctly associated with the intended keys.
    """
    
    standard_dict = {}
    keywords = [HDU_keyword, f"{HDU_keyword}_MASK", f"{HDU_keyword}_ERROR", f"{HDU_keyword}_FLAG"]
    flags = ["data","mask","err","qual"]

    for keyword, mapdata, flag in zip(keywords, mapdict.values(), flags):
        standard_header_dict = standard_header_dict(galname, HDU_keyword, unit_str, flag)
        standard_dict[keyword] = (mapdata, standard_header_dict)

    return standard_dict


def header_dict_formatter(header_dict):
    """
    ...

    Parameters
    ----------
    header_dict : dict
        Dictionary of {str: tuple} to be written to the header of the
        numpy.ndarray image where the keys are strings of the header keywords and the tuple
        contains the value and comment to be assigned to the header keywords.
        The tuple values of `header_dict` should be structured 
        `(value, comment)` where
            - `value` : An object containing the value to be placed in the header.
            - `comment` : A str of the comment describing the value.
        See `astropy.io.fits` FITS Headers Documentation 
        <https://docs.astropy.org/en/latest/io/fits/usage/headers.html> for additional 
        information.

    Returns
    -------
    dict
        Returns the header_dict with any necessary changes for formatting FITS headers. Returns
        exactly header_dict if no reformatting is necessary.
    """

    def card_builder(key,value,comment):
        return f"{key.ljust(8)}={str(value).rjust(21)} / {comment}"

    reformat_dict = {}

    for key, header_instance in header_dict.items():
        if len(key)>8:
            key = key[:8]

        if not isinstance(header_instance, tuple):
            header_instance = (header_instance, '')

        value = header_instance[0]
        comment = header_instance[1]
        
        card = card_builder(key,value,comment)

        if len(card)>80:
            split_comment = card[:80].split('/ ')[1]
            remainder = card[80:]
            reformat_dict[key] = (value, split_comment)
            reformat_dict['COMMENT'] = remainder

        else:
            reformat_dict[key] = (value, comment)

    return reformat_dict


def map_file_handler(galdir, maps_dict, filepath = None, overwrite = True, verbose = False):
    """
    Update or create a FITS file with specified image HDUs and optional headers.
    
    This function checks if a FITS file at the given `filepath` exists.
    If the file exists, it updates or appends `ImageHDU`s for each array
    in `maps_dict`. If an HDU with the same name as a key in `maps_dict`
    already exists in the file, the function overwrites its data and updates
    its header. If no HDU with that name exists, it appends a new `ImageHDU`
    with the given name, data, and header. If the file does not exist, it creates
    a new FITS file with `ImageHDU`s for each array in `maps_dict`.

    Parameters
    ----------
    galdir : str
        String specifying the specific galaxy and binning method directory to 
        write the data into; should be formatted as `"{galaxy_name}-{binning_method}"`

    maps_dict : dict of {str: (numpy.ndarray, dict)}
        Dictionary where each key is a string representing the HDU name,
        each value is a tuple containing:
            - a 2D `numpy.ndarray` representing the image data, and
            - A dictionary of {str: tuple} to be written to the header of the
            numpy.ndarray image. See `astropy.io.fits` [FITS Headers Documentation](https://docs.astropy.org/en/latest/io/fits/usage/headers.html) for 
            additional information.

    filepath : str, optional
        Optionally specify a full filepath of the FITS file to be written including the filename.
    
    overwrite : bool, optional
        Default `True` specifying to write into the already existing FITS file if it exists,
        and overwrite existing `maps_dict` data within the file. If `False`, a new FITS 
        file containing the `maps_dict` data will be written with an integer identifier in 
        the filename unless `filepath` is specified and the `filepath` file does not already
        exist.

    verbose : bool, optional
        ...

    Raises
    ------
    ValueError
        If any array in `maps_dict` is not a 2-dimensional `numpy.ndarray`,
        a `ValueError` is raised.
    
    Notes
    -----
    - If the FITS file exists, this function updates it in place. HDUs with
      matching names have their `.data` attribute overwritten and headers
      updated with the provided dictionary.
    - If the FITS file does not exist, a new file is created containing all
      `ImageHDU`s in `maps_dict`, plus a default `PrimaryHDU`.
    - FITS files allow HDUs with the same name, but this function only overwrites
      the first HDU with a matching name if it exists, and appends any others as new HDUs.

    Examples
    --------
    Create or update a FITS file with two image HDUs and custom headers:

    >>> import numpy as np
    >>> maps_dict = {
    ...     'Image1': (np.random.random((100, 100)), {'EXPOSURE': 30}),
    ...     'Image2': (np.random.random((200, 200)), {'EXPOSURE': (60, 'The exposure time in seconds'), 'FILTER': 'V'})
    ... }
    >>> map_file_handler('example.fits', maps_dict)

    This will either create 'example.fits' with two `ImageHDU`s named 'Image1'
    and 'Image2' with the specified headers, or update any existing HDUs with
    the same names and update their headers.

    """

    if filepath is None:
        rootpath = defaults.get_default_path()
        base_name = f"{galdir}-local-MAPS.fits"
        filepath = os.path.join(os.path.dirname(rootpath),"data/local_outputs", galdir, base_name)
    
    full_path = os.path.dirname(filepath)
    file_name = os.path.basename(filepath)
    util.check_filepath(full_path, mkdir = True, verbose = verbose, error = True)

    # if overwrite and the FITS file exists
    if overwrite and os.path.exists(filepath):
        # Open the existing file in update mode
        with fits.open(filepath, mode='update') as hdul:
            # Track existing HDU names
            existing_names = {hdu.name for hdu in hdul if isinstance(hdu, fits.ImageHDU)}
            i=0
            for name, (data, header_dict) in maps_dict.items():
                header_dict = header_dict_formatter(header_dict)
                if not isinstance(data, np.ndarray) or data.ndim != 2:
                    raise ValueError(f"Data for '{name}' must be a 2-dimensional numpy array.")
                
                if name in existing_names:
                    # Overwrite data for existing HDU with this name
                    hdu = hdul[name]
                    hdu.data = data
                    
                    # Update the header with the provided header dictionary
                    for key, value in header_dict.items():
                        hdu.header[key] = value

                else:
                    # Append a new ImageHDU if it doesn't already exist
                    new_hdu = fits.ImageHDU(data=data, name=name)
                    for key, value in header_dict.items():
                        new_hdu.header[key] = value
                    hdul.append(new_hdu)


                dots = '.' * ((i % 3) + 1)
                print(f"Writing data to {filepath} {i+1}/{len(data)}{dots}", end='\r')
                i+=1
            hdul.flush()  # Save changes
            print("Done!       ")
            print(f"Data saved to {filepath}")
    else:
        # Create a new FITS file with the given HDUs
        hdul = fits.HDUList([fits.PrimaryHDU()])  # Start with a Primary HDU
        i=0
        for name, (data, header_dict) in maps_dict.items():
            header_dict = header_dict_formatter(header_dict)
            if not isinstance(data, np.ndarray) or data.ndim != 2:
                raise ValueError(f"Data for '{name}' must be a 2-dimensional numpy array.")
            
            new_hdu = fits.ImageHDU(data=data, name=name)
            # Add header information if provided
            for key, value in header_dict.items():
                new_hdu.header[key] = value
            hdul.append(new_hdu)

            dots = '.' * ((i % 3) + 1)
            print(f"Writing data to new FITS file {i+1}/{len(data)}{dots}", end='\r')
            i+=1
        print("Done!     ")
        
        if os.path.exists(filepath):
            fileslist = glob(os.path.join(full_path, "*.fits"))
            n = len(fileslist)
            padded_number = str(n).zfill(3)
            file_name += padded_number
            filepath = os.path.join(full_path, file_name)

        hdul.writeto(filepath, overwrite=True)
        print(f"Created new FITS file '{filepath}'.")

# 
# 
# Configuration Files Section
# Parser and cleaner for reading .ini files
# 
# 

def parse_config(config_fil, verbose):
    """
    Parses a configuration file to retrieve data from the 'default' section.

    This function reads a configuration file (typically in `.ini` format) and attempts to parse
    it using `configparser`. If a parsing error occurs, the function applies a cleaning step
    via `modules.file_handler.clean_ini_file` and retries until successful.

    Parameters
    ----------
    config_fil : str
        Path to the configuration file to be parsed.
    verbose : bool
        If True, outputs detailed messages about any errors encountered and steps taken
        during file cleaning.

    Returns
    -------
    configparser.SectionProxy
        The 'default' section of the configuration file as a `configparser.SectionProxy` object,
        allowing access to key-value pairs within this section.

    Notes
    -----
    The function repeatedly attempts to read the configuration file until successful. If a
    `configparser.Error` is raised, it cleans the file by calling `clean_ini_file` from
    `modules.util`, overwriting the file if necessary, and attempts to parse it again. Verbose
    output provides information on errors and cleaning actions when `verbose` is set to True.

    Example
    -------
    >>> config_data = parse_config("config.ini", verbose=True)
    >>> print(config_data["z"])
    '0.023'
    """

    config = configparser.ConfigParser()
    parsing = True
    while parsing:
        try:
            config.read(config_fil)
            parsing = False
        except configparser.Error as e:
            util.verbose_print(verbose, f"Error parsing file: {e}")
            util.verbose_print(verbose, f"Cleaning {config_fil}")
            clean_ini_file(config_fil, overwrite=True)


    return config['default']

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