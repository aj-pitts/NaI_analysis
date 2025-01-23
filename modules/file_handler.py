from astropy.io import fits
import os
import re
from datetime import datetime
from glob import glob
import numpy as np
import modules.defaults as defaults
import modules.util as util
import warnings
import configparser
from datetime import datetime
from modules.util import verbose_print

# TODO: update the docstring, add verbose printing, add better bc/nc option

# 
# 
# Data acquisition
# 
# 

def init_datapaths(galname, bin_method, verbose=True):
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

        • 'CONFIG' (str): Path to the `.ini` configuration file.

        • 'NO-CORR' (dict): Dictionary with filepaths for non-corrected data:
        
            - 'LOGCUBE' (str): Path to the non-corrected logcube file.
            - 'MAPS' (str): Path to the non-corrected maps file.

        • 'BETA-CORR' (dict): Dictionary with filepaths for beta-corrected data:

            - 'LOGCUBE' (str): Path to the beta-corrected logcube file.
            - 'MAPS' (str): Path to the beta-corrected maps file.
            - 'MCMC' (list): List of paths to MCMC results.
            - 'LOCAL' (str): Path to the locally corrected data.

        **Note**: If a file does not exist, its value in the dictionary will be `None`.
    """

    verbose_print(verbose, f"Acquiring relevant files for {galname}-{bin_method}")
    ## initialize paths to the data directory and relevant subdirectories
    pipeline_data = defaults.get_data_path(subdir='pipeline')
    local_data = defaults.get_data_path(subdir='local')


    galsubdir = f"{galname}-{bin_method}"
    analysisplan = defaults.analysis_plans()
    corr_key = 'BETA-CORR'

    dappath = os.path.join(pipeline_data, "dap_outputs", galsubdir, corr_key, f"{bin_method}-{analysisplan}")
    mcmcpath = os.path.join(pipeline_data, "mcmc_outputs", galsubdir, corr_key, analysisplan)
    musecubepath = os.path.join(pipeline_data, "muse_cubes", galname)
    localdatapath = os.path.join(local_data, "local_outputs", galsubdir, corr_key, analysisplan)

    outdict = {'CONFIG':None, 
               'LOGCUBE':None, 
               'MAPS':None, 
               'MCMC':None, 
               'LOCAL':None}
    
    ### CONFIG file
    configfils = glob(os.path.join(musecubepath, "*.ini"))
    if len(configfils) == 0:
        util.verbose_warning(verbose,f"No configuration file found in {musecubepath}")

    else:
        if len(configfils)>1:
            warnings.warn(f"{musecubepath} has more than one config file:\n{configfils} \nDefaulting to {configfils[0]}")

        outdict['CONFIG'] = configfils[0]
        verbose_print(verbose, f"Using CONFIG File: {configfils[0]}")

    ### DAP files
    if not os.path.exists(dappath):
        util.verbose_warning(verbose,f"Filepath does not exist: {dappath}")
    else:
        beta_corr_fils = glob(os.path.join(dappath, "**", "*.fits"), recursive=True)
        if len(beta_corr_fils) != 0:
            for fil in beta_corr_fils:
                if 'LOGCUBE' in fil:
                    outdict['LOGCUBE'] = fil
                    verbose_print(verbose, f"Using LOGCUBE File: {fil}")
                if 'MAPS' in fil:
                    outdict['MAPS'] = fil
                    verbose_print(verbose, f"Using MAPS File: {fil}")
        else:
            util.verbose_warning(verbose, f"No LOGCUBE or MAPS found in {dappath}")


    ### MCMC files
    if not os.path.exists(mcmcpath):
        util.verbose_warning(verbose,f"Filepath does not exist: {mcmcpath}")
    else:
        mcmc_subdirs = [d for d in os.listdir(mcmcpath) if os.path.isdir(os.path.join(mcmcpath, d))]
        if len(mcmc_subdirs)==0:
            util.verbose_warning(verbose, f"Empty directory: {mcmcpath}")
        
        elif len(mcmc_subdirs)==1:
            mcmc_rundir = os.path.join(mcmcpath, mcmc_subdirs[0])
            mcmc_files = glob(os.path.join(mcmc_rundir, "*.fits"))
            verbose_print(verbose, f"Using {len(mcmc_files)} MCMC Files found in: {mcmc_rundir}")

        else:
            dated_dirs = []
            pattern = re.compile(r"^Run_(\d{4}-\d{2}-\d{2})$")
            for subdir in mcmc_subdirs:
                match = pattern.match(subdir)
                if match:
                    date_str = match.group(1)
                    try:
                        date = datetime.strptime(date_str, "%Y-%m-%d")
                        dated_dirs.append((subdir, date))
                    except ValueError:
                        pass  # Ignore invalid date formats

            # Find the subdirectory with the most recent date
            if dated_dirs:
                most_recent_dir = max(dated_dirs, key=lambda x: x[1])
                print("Most recent subdirectory:", most_recent_dir[0])
                mcmc_rundir = os.path.join(mcmcpath, most_recent_dir[0])
                mcmc_files = glob(os.path.join(mcmc_rundir, "*.fits"))
                verbose_print(verbose, f"Using {len(mcmc_files)} MCMC Files found in: {mcmc_rundir}")
            else:
                print("No valid 'Run_YYYY-MM-DD' subdirectories found.")

        outdict['MCMC'] = mcmc_files



    ### LOCAL file
    if not os.path.exists(localdatapath):
        util.verbose_warning(verbose,f"Filepath does not exist: {localdatapath}")
    else:
        localfils = glob(os.path.join(localdatapath, "*.fits"))
        if len(localfils) == 0:
            util.verbose_warning(verbose,f"No local data file found in {localdatapath}")
        else:
            outdict['LOCAL'] = localfils
            for file in localfils:
                verbose_print(verbose, f"Using LOCAL File: {file}")

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
      Includes fields like "DESC", "BUNIT", "ERRDATA", "QUALDATA", "EXTNAME", and "AUTHOR".
    
    - **For "err"**:
      Includes fields like "BUNIT", "DATA", "QUALDATA", "EXTNAME", and "AUTHOR".
    
    - **For "mask"**:
      Includes fields like "ERRDATA", "EXTNAME", and "AUTHOR".

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
            "EXTNAME":(HDU_keyword, "extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    
    elif flag == "err":
        header = {
            "BUNIT":(unit_str, "Unit of pixel value"),
            "DATA":(HDU_keyword, "Associated data extension"),
            "QUALDATA":(f"{HDU_keyword}_MASK", "Associated quality extension"),
            "EXTNAME":(f"{HDU_keyword}_ERR", "extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    
    elif flag == "mask":
        quality = defaults.local_quality_flag()
        header = {
            "DATA":("","Data Quality Bitmask"),
            "Bit_0":(quality[0],"Description of bitmask 0"),
            "Bit_1":(quality[1],"Description of bitmask 1"),
            "Bit_2":(quality[2],"Description of bitmask 2"),
            "Bit_3":(quality[3],"Description of bitmask 3"),
            "Bit_4":(quality[4],"Description of bitmask 4"),
            "Bit_5":(quality[5],"Description of bitmask 5"),
            "Bit_6":(quality[6],"Description of bitmask 6"),
            "Bit_7":(quality[7],"Description of bitmask 6"),
            "ERRDATA":(f"{HDU_keyword}_ERR", "Associated uncertainty values extension"),
            "EXTNAME":(f"{HDU_keyword}_MASK", "extension name"),
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
    keywords = [HDU_keyword, f"{HDU_keyword}_MASK", f"{HDU_keyword}_ERROR"]
    flags = ["data","mask","err"]

    for keyword, mapdata, flag in zip(keywords, mapdict.values(), flags):
        standard_header = standard_header_dict(galname, HDU_keyword, unit_str, flag)
        standard_dict[keyword] = (mapdata, standard_header)

    return standard_dict


def header_dict_formatter(header_dict):
    """
    Formats a dictionary of FITS header keywords, ensuring each card adheres to the 80-character limit.

    Parameters
    ----------
    header_dict : dict
        Dictionary of {str: tuple} where:
        - `key`: str, FITS keyword (max 8 characters).
        - `value`: object, value associated with the keyword.
        - `comment`: str, description/comment about the value.

    Returns
    -------
    dict
        A formatted dictionary ready for FITS headers, ensuring compliance with the 80-character rule.
    """

    def card_builder(key, value, comment):
        """
        Build an initial FITS header card string.
        """
        return f"{key[:8].ljust(8)}= {str(value).rjust(20)} / {comment}"

    reformat_dict = {}

    for key, header_instance in header_dict.items():
        # Ensure the key is max 8 characters
        key = key[:8]

        if not isinstance(header_instance, tuple):
            header_instance = (header_instance, '')

        value, comment = header_instance

        # Build the initial card
        card = card_builder(key, value, comment)

        if len(card) <= 80:
            # If card fits within 80 chars, keep it as is
            reformat_dict[key] = (value, comment)
        else:
            # Split into key=value and comment
            key_value_part = f"{key[:8].ljust(8)}= {str(value).rjust(20)}"
            remaining_comment = f" / {comment}"
            
            # Handle first 80 characters
            first_line = (key_value_part + remaining_comment)[:80]
            reformat_dict[key] = (value, first_line.split('/ ', 1)[-1])
            
            # Handle overflow in comments
            overflow_comment = remaining_comment[len(first_line) - len(key_value_part):].strip()
            while len(overflow_comment) > 0:
                next_chunk = overflow_comment[:72]  # Reserve space for "COMMENT "
                overflow_comment = overflow_comment[72:]
                reformat_dict['COMMENT'] = next_chunk.strip()

    return reformat_dict


def simple_file_handler(galdir, maps_dict, filename, filepath, overwrite = True, verbose = False):
    util.check_filepath(filepath, mkdir=True, verbose=verbose, error=True)

    filename = f"{galdir}-{filename}.fits"
    full_path = os.path.join(filepath, filename)

    if os.path.isfile(full_path):
        if overwrite:
            os.remove(full_path)
        else:
            current_files = [f for f in os.listdir(filepath) if not f.startswith('.')]
            n = len(current_files)
            padded_number = str(n).zfill(3)
            base_name, ext = os.path.splitext(filename)
            filename = f"{base_name}-{padded_number}{ext}"
            full_path = os.path.join(filepath, filename)
    

    hdul = fits.HDUList([fits.PrimaryHDU()])  # Start with a Primary HDU
    for name, (data, header_dict) in maps_dict.items():
        header_dict = header_dict_formatter(header_dict)
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError(f"Data for '{name}' must be a 2-dimensional numpy array.")
        
        new_hdu = fits.ImageHDU(data=data, name=name)
        # Add header information if provided
        for key, value in header_dict.items():
            new_hdu.header[key] = value
        hdul.append(new_hdu)

    hdul.writeto(full_path, overwrite=True)
    print(f"Created new FITS file '{full_path}'")


def map_file_handler(galdir, maps_dict, filepath, verbose = False):
    """
    Handles the creation and updating of a FITS file containing galaxy map data.

    This function creates or updates a FITS file named `<galdir>-local-maps.fits` 
    in the specified `filepath`. If the file does not exist, it initializes a new
    FITS file with the data provided in `maps_dict`. If the file exists, it updates
    the data for existing extensions or adds new extensions as needed. The function
    also timestamps the header of updated or newly added extensions.

    Parameters
    ----------
    galdir : str
        The name of the galaxy directory, standard format of `<galaxy_name>-<bin_method>`;
        used to construct the FITS filename.
    maps_dict : dict
        A dictionary containing the map data and header information. 
        Keys are extension names (EXTNAME), and values are tuples of the form 
        `(data, header_dict)`, where:
        - `data` is a 2D NumPy array containing the map data.
        - `header_dict` is a dictionary of header keyword-value pairs.
    filepath : str
        The directory where the FITS file should be created or updated.
    verbose : bool, optional
        If True, prints messages about the process (default is False).

    Notes
    -----
    - The function ensures the FITS file contains a Primary HDU.
    - The `UPDATE` keyword in the header is used to record the timestamp 
      of the last modification for updated or new extensions.
    - Duplicate extension names in `maps_dict` are not allowed.

    Examples
    --------
    >>> galdir = "galaxy1"
    >>> maps_dict = {
    ...     "EXT1": (np.random.random((100, 100)), {"KEY1": "VALUE1"}),
    ...     "EXT2": (np.random.random((200, 200)), {"KEY2": "VALUE2"})
    ... }
    >>> filepath = "/path/to/directory"
    >>> map_file_handler(galdir, maps_dict, filepath, verbose=True)
    Created new FITS file: /path/to/directory/galaxy1-local-maps.fits

    Raises
    ------
    ValueError
        If `maps_dict` is empty or contains duplicate extension names.
    """

    # check if the filepath exists
    util.check_filepath(filepath, mkdir=True, verbose=verbose, error=True)

    # init the filename and fullpath to the output file
    file_name = f"{galdir}-local_maps.fits"
    full_path = os.path.join(filepath, file_name)

    # timestamp for the header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # init a new HDU list and write the maps data into it
    new_hdul = fits.HDUList([])

    for name, (data, header_dict) in maps_dict.items():
        header_dict_formatted = header_dict_formatter(header_dict=header_dict)
        image_hdu = fits.ImageHDU(data=data, name=name)

        for key, value in header_dict_formatted.items():
            image_hdu.header[key] = value

        image_hdu.header['UPDATED'] = (timestamp, "Last updated timestamp")
        new_hdul.append(image_hdu)



    # if the file is not already made, write it with the mapsdict data
    if not os.path.isfile(full_path):
        new_hdul.insert(0, fits.PrimaryHDU())

        util.verbose_print(verbose, f"Writing data to new file: {full_path}")
        new_hdul.writeto(full_path)

        util.verbose_print(verbose, "Done.")
    
    # if the file exists, open it in update mode
    else:
        util.verbose_print(verbose, f"Found file: {full_path}")
        with fits.open(full_path, mode='update') as existing_hdul:
            util.verbose_print(verbose, f"Updating...")
            # grab the extnames from the new hdu
            existing_names = [hdu.name for hdu in existing_hdul]

            existing_name_output = []
            new_name_output = []
            # iterate through the new HDU list.
            for new_hdu in new_hdul:
                extname = new_hdu.name

                # If an HDU in the new list matches the existing data names, update the existing data with the new data
                if extname in existing_names:
                    existing_name_output.append(extname)
                    existing_hdul[extname].data = new_hdu.data
                    existing_hdul[extname].header['UPDATED'] = (timestamp, "Last updated timestamp")
                
                # otherwise, append the new data into the file
                else:
                    new_name_output.append(extname)
                    existing_hdul.append(new_hdu)

            util.verbose_print(verbose, f"Updating file: {full_path}")
            existing_hdul.flush(verbose=verbose)
            if len(existing_name_output)>0:
                util.verbose_print(verbose, f"Updated HDU Images: {existing_name_output}")
            if len(new_name_output)>0:
                util.verbose_print(verbose, f"Added HDU Images: {new_name_output}")
            util.verbose_print(verbose, "Done.")
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




### old location map_file_handler
# if name in existing_names:
#     # Overwrite data for existing HDU with this name
#     hdu = hdul[name]
#     hdu.data = data
#     # Update the header with the provided header dictionary
#     for key, value in header_dict.items():
#         hdu.header[key] = value
# else:
#     # Append a new ImageHDU if it doesn't already exist
#     new_hdu = fits.ImageHDU(data=data, name=name)
#     for key, value in header_dict.items():
#         new_hdu.header[key] = value
#     hdul.append(new_hdu)


# def header_dict_formatter_old(header_dict):

#     def card_builder(key,value,comment):
#         return f"{key.ljust(8)}={str(value).rjust(21)} / {comment}"

#     reformat_dict = {}

#     for key, header_instance in header_dict.items():
#         if len(key)>8:
#             key = key[:8]

#         if not isinstance(header_instance, tuple):
#             header_instance = (header_instance, '')

#         value = header_instance[0]
#         comment = header_instance[1]
        
#         card = card_builder(key,value,comment)

#         if len(card)>80:
#             split_card = card[:80].split('/ ',)
#             split_comment = split_card[0]
#             remainder = card[80:]
#             reformat_dict[key] = (value, split_comment)
#             reformat_dict['COMMENT'] = remainder

#         else:
#             reformat_dict[key] = (value, comment)

#     return reformat_dict




# def map_file_handler_old(galdir, maps_dict, filepath, overwrite = True, verbose = False):

#     util.check_filepath(filepath, mkdir=True, verbose=verbose, error=True)

#     file_name = f"{galdir}-local-maps.fits"
#     full_path = os.path.join(filepath, file_name)

#     # if overwrite and the FITS file exists
#     if overwrite and os.path.isfile(full_path):
#         # Open the existing file in update mode
#         with fits.open(full_path, mode='update') as hdul:
#             new_hdul = fits.HDUList()

#             # Track existing HDU names
#             existing_names = {hdu.name for hdu in hdul if isinstance(hdu, fits.ImageHDU)}
#             i=0

#             for name, (data, header_dict) in maps_dict.items():
#                 header_dict = header_dict_formatter(header_dict)
#                 if not isinstance(data, np.ndarray) or data.ndim != 2:
#                     raise ValueError(f"Data for '{name}' must be a 2-dimensional numpy array.")

#                 # Create a new HDUList preserving order and replacing matching HDUs
#                 new_hdul = fits.HDUList()

#                 for hdu in hdul:
#                     if isinstance(hdu, fits.ImageHDU) and hdu.name == name:
#                         # Replace the HDU with the new one
#                         updated_hdu = fits.ImageHDU(data=data, name=name)
#                         for key, value in header_dict.items():
#                             updated_hdu.header[key] = value
#                         new_hdul.append(updated_hdu)
#                     else:
#                         # Keep the existing HDU unchanged
#                         new_hdul.append(hdu)

#                 # Clear the original HDUList and replace it with the new one
#                 hdul.clear()
#                 hdul.extend(new_hdul)

#                 print(f"Writing data to HDUL...")
#             hdul.flush()  # Save changes
#             print("Done!       ")
#             print(f"Data saved to {full_path}")
#     else:
#         # Create a new FITS file with the given HDUs
#         hdul = fits.HDUList([fits.PrimaryHDU()])  # Start with a Primary HDU
#         for name, (data, header_dict) in maps_dict.items():
#             header_dict = header_dict_formatter(header_dict)
#             if not isinstance(data, np.ndarray) or data.ndim != 2:
#                 raise ValueError(f"Data for '{name}' must be a 2-dimensional numpy array.")
            
#             new_hdu = fits.ImageHDU(data=data, name=name)
#             # Add header information if provided
#             for key, value in header_dict.items():
#                 new_hdu.header[key] = value
#             hdul.append(new_hdu)

#         current_files = [f for f in os.listdir(filepath) if not f.startswith('.')]
#         if len(current_files)>0:
#             fileslist = glob(os.path.join(filepath, "*.fits"))
#             n = len(fileslist)
#             padded_number = str(n).zfill(3)
#             base_name, ext = os.path.splitext(file_name)
#             file_name = f"{base_name}-{padded_number}{ext}"
#             full_path = os.path.join(filepath, file_name)

#         hdul.writeto(full_path, overwrite=True)
#         print(f"Created new FITS file '{filepath}'.")