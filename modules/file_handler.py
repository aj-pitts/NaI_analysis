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
from tqdm import tqdm
import yaml

# TODO: update the docstring, add verbose printing, add better bc/nc option

# 
# 
# Data acquisition
# 
# 

def init_datapaths(galname, bin_method, verbose=False, redshift = True):
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
    localdatasubpath = os.path.join(localdatapath, "maps")

    outdict = {'CONFIG':None, 
               'LOGCUBE':None, 
               'MAPS':None, 
               'MCMC':None, 
               'LOCAL':None,
               'LOCAL_MAPS':None}
    
    ### CONFIG file
    configfils = glob(os.path.join(musecubepath, "*.ini"))
    if len(configfils) == 0:
        util.verbose_warning(verbose,f"No configuration file found in {musecubepath}")

    else:
        if len(configfils)>1:
            warnings.warn(f"{musecubepath} has more than one config file:\n{configfils} \nDefaulting to {configfils[0]}")

        outdict['CONFIG'] = configfils[0]
        verbose_print(verbose, f"'CONFIG' key: {configfils[0]}")
        
    if redshift:
        configuration = parse_config(configfils[0], verbose=verbose)
        redshift = configuration['z']
        outdict['Z'] = float(redshift)
        #verbose_print(verbose, f"Redshift z = {redshift} found in {configfils[0]}")
        verbose_print(verbose, f"'Z' key: {redshift}")

    ### DAP files
    if not os.path.exists(dappath):
        util.verbose_warning(verbose,f"Filepath does not exist: {dappath}")
    else:
        beta_corr_fils = glob(os.path.join(dappath, "**", "*.fits"), recursive=True)
        if len(beta_corr_fils) != 0:
            for fil in beta_corr_fils:
                if 'LOGCUBE' in fil:
                    outdict['LOGCUBE'] = fil
                    verbose_print(verbose, f"'LOGCUBE' key: {fil}")
                if 'MAPS' in fil:
                    outdict['MAPS'] = fil
                    verbose_print(verbose, f"'MAPS' key: {fil}")
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
            verbose_print(verbose, f"'MCMC' key: {len(mcmc_files)} MCMC Files found in {mcmc_rundir}")

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
                verbose_print(verbose, "Most recent subdirectory:", most_recent_dir[0])
                mcmc_rundir = os.path.join(mcmcpath, most_recent_dir[0])
                mcmc_files = glob(os.path.join(mcmc_rundir, "*.fits"))
                verbose_print(verbose, f"'MCMC' key: {len(mcmc_files)} MCMC Files found in {mcmc_rundir}")
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
            for file in localfils:
                if "local_maps" in file:
                    outdict['LOCAL'] = file
                    verbose_print(verbose, f"Using LOCAL File: {file}")
        if outdict['LOCAL'] is None:
            util.verbose_warning(verbose, f"*local_maps.fits file not found in {localdatapath}")

    ### individual local maps
    if not os.path.exists(localdatasubpath):
        util.verbose_warning(verbose,f"Filepath does not exist: {localdatasubpath}")
    else:
        localfils = glob(os.path.join(localdatasubpath, "*.fits"))
        if len(localfils) == 0:
            util.verbose_warning(verbose,f"No map files in {localdatasubpath}")
        else:
            outdict['LOCAL_MAPS'] = localfils
            verbose_print(f"'LOCAL' key: {len(localfils)} local maps found in {localdatasubpath}")

    return outdict

# 
# 
# Section for handling FITS file writing
# format headers, automatically write 2D maps into FITS Files
# 
# 

def standard_header_dict(galname, HDU_keyword, unit_str, flag, add_descrip = None, additional_mask_bits = None, 
                         asymmetric_error = False):
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
            "DESC":(f"{galname} {HDU_keyword.replace("_"," ")} map",""),
            "BUNIT":(unit_str, "Unit of pixel value"),
            "ERRDATA":(f"{HDU_keyword}_ERROR", "Associated uncertainty values extension"),
            "QUALDATA":(f"{HDU_keyword}_MASK", "Associated quality extension"),
            "EXTNAME":(HDU_keyword, "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    
    elif flag == "error":
        if asymmetric_error:
            header = {
                "DESC":(f"Assymetric uncertainty map for {galname} {HDU_keyword.replace("_"," ")}", ""),
                "BUNIT":(unit_str, "Unit of pixel value"),
                "C01":("Error propagated from MCMC 16th percentile", "Data in Channel 1"),
                "C02":("Error propagated from MCMC 84th percentile", "Data in Channel 2"),
                "DATA":(HDU_keyword, "Associated data extension"),
                "QUALDATA":(f"{HDU_keyword}_MASK", "Associated quality extension"),
                "EXTNAME":(f"{HDU_keyword}", "Extension name"),
                "AUTHOR":("Andrew Pitts","")
            }
        else:
            header = {
                "DESC":(f"Uncertainty map for {galname} {HDU_keyword.replace("_"," ")}", ""),
                "BUNIT":(unit_str, "Unit of pixel value"),
                "DATA":(HDU_keyword, "Associated data extension"),
                "QUALDATA":(f"{HDU_keyword}_MASK", "Associated quality extension"),
                "EXTNAME":(f"{HDU_keyword}", "Extension name"),
                "AUTHOR":("Andrew Pitts","")
            }
    
    elif flag == "mask":
        quality = defaults.local_quality_flag()
        header = {
            "DATA":("","Data Quality Categorical Mask"),
            "Int_0":(quality[0],"Description of mask int 0"),
            "Int_1":(quality[1],"Description of mask int 1"),
            "Int_2":(quality[2],"Description of mask int 2"),
            "Int_3":(quality[3],"Description of mask int 3"),
            "Int_4":(quality[4],"Description of mask int 4"),
            "Int_5":(quality[5],"Description of mask int 5"),
            "Int_6":(quality[6],"Description of mask int 6"),
            "Int_7":(quality[7],"Description of mask int 7"),
            "Int_8":(quality[8],"Description of mask int 8"),
            "ERRDATA":(f"{HDU_keyword}_ERROR", "Associated uncertainty values extension"),
            "EXTNAME":(f"{HDU_keyword}", "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
        if additional_mask_bits is not None:
            occupied = np.arange(9)
            items = list(header.items())
            
            for i in range(len(additional_mask_bits)):
                index = i+10
                new_int = additional_mask_bits[i][0]
                new_int_desc = additional_mask_bits[i][1]

                if new_int in occupied:
                    raise ValueError(f"Additional Int Mask {str(new_int)} is already used by quality flag.")
                
                bit_str = f'Bit_{str(new_int)}'

                items.insert(index, (bit_str, (new_int_desc, f'Description of mask int {str(new_int)}')))
            
            header = items.dict()


    elif flag == "additional":
        header = {
            "DESC":(f"{galname} {add_descrip}",""),
            "BUNIT":(unit_str, "Unit of pixel value"),
            "QUALDATA":(f"{HDU_keyword}_MASK", "Associated quality extension"),
            "EXTNAME":(HDU_keyword, "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    
    return header


def standard_map_dict(galname, mapdict, HDU_keyword = None, IMAGE_units = None, additional_keywords = [], additional_units = [],
                    additional_descriptions = [], additional_mask_bits = [], custom_header_dict = None, asymmetric_error = False):
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

    The header dictionary is created using the `standard_header_dict` function, which includes the
    galaxy name, HDU keyword, unit string, and a specific flag for each type of data.

    **Important**: The `mapdict` must be ordered correctly, as the order of the values in `mapdict` 
    directly determines the keys of the output dictionary. If the order is incorrect, the corresponding 
    data may not be correctly associated with the intended keys.
    """
    
    standard_dict = {}

    for param in [additional_descriptions, additional_keywords, additional_units]:
        if not isinstance(param, list):
            raise ValueError(f"Parameters additional_keywords, additional_units, and additional_descriptions must be Python lists.")

    if custom_header_dict is not None:
        for keyword, mapdata, header_dict in zip(custom_header_dict.keys(), mapdict.values(), custom_header_dict.values()):
            standard_dict[keyword] = (mapdata, header_dict)

    else:
        if HDU_keyword is None or IMAGE_units is None:
            raise ValueError("HDU_keyword and IMAGE_units parameters required if custom_header_dict is None.")
        
        keywords = [HDU_keyword] + additional_keywords + [f"{HDU_keyword}_MASK", f"{HDU_keyword}_ERROR"]
        unit_list = [IMAGE_units] + additional_units + ['', IMAGE_units]
        descriptions = [''] + additional_descriptions + ['', '']

        flags = ["data"] + ['additional'] * len(additional_keywords) + ["mask", "error"]

        if len(keywords) != len(unit_list):
            raise ValueError(f"additional_keywords and additional_units must have the same length")
        
        for keyword, mapdata, units, descrip, flag in zip(keywords, mapdict.values(), unit_list, descriptions, flags):
            if keyword == 'mask':
                standard_header = standard_header_dict(galname, keyword, units, flag, add_descrip=descrip, 
                                                       additional_mask_bits=additional_mask_bits)
            else:
                standard_header = standard_header_dict(galname, keyword, units, flag, add_descrip=descrip, 
                                                       asymmetric_error=asymmetric_error)
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

def write_maps_file(
    galname,
    bin_method,
    maps_dict_list,
    verbose=False,
    preserve_standard_order=True,
):
    """
    Incrementally updates or creates a FITS file by writing maps one at a time.

    Parameters
    ----------
    galdir : str
        Galaxy name prefix.
    maps_dict_list : list of dict
        Each dict maps EXTNAME to (data, header_dict)
    filepath : str
        Directory for FITS file.
    verbose : bool
        Print verbose messages.
    preserve_standard_order : bool
        Whether to reorder extensions in standard order.

    """
    galdir = f'{galname}-{bin_method}'

    # build the path
    corr_key = 'BETA-CORR'
    analysis_plan = defaults.analysis_plans()
    local_data = defaults.get_data_path(subdir='local')
    filepath = os.path.join(local_data, 'local_outputs', f"{galname}-{bin_method}", corr_key, analysis_plan)
    util.check_filepath(filepath, mkdir=True,verbose=verbose)

    file_name = f"{galdir}-local_maps.fits"
    full_path = os.path.join(filepath, file_name)

    # timestamp for updates
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # If file exists, open it
    if os.path.isfile(full_path):
        hdul = fits.open(full_path, mode='update')
        existing_names = [hdu.name for hdu in hdul]
        if verbose:
            print(f"Updating existing FITS file: {full_path}")
    else:
        # create a new FITS file with PrimaryHDU
        hdul = fits.HDUList([fits.PrimaryHDU()])
        existing_names = []
        if verbose:
            print(f"Creating new FITS file: {full_path}")

    # Process each map dictionary in the list
    for maps_dict in maps_dict_list:
        for extname, (data, header_dict) in maps_dict.items():
            header_dict_formatted = header_dict_formatter(header_dict=header_dict)

            # create new ImageHDU
            new_hdu = fits.ImageHDU(data=data, name=extname)
            for key, value in header_dict_formatted.items():
                new_hdu.header[key] = value
            new_hdu.header['UPDATED'] = (timestamp, "Last updated timestamp")

            if extname in existing_names:
                # Replace existing HDU
                index = existing_names.index(extname)
                if verbose:
                    print(f"Overwriting HDU: {extname}")
                hdul[index] = new_hdu
            else:
                # Append new HDU
                if verbose:
                    print(f"Adding new HDU: {extname}")
                hdul.append(new_hdu)
                existing_names.append(extname)

    # Reorder if needed
    if preserve_standard_order:
        hdul = reorder_hdu(hdul)

    # Write changes
    hdul.writeto(full_path, overwrite=True)
    hdul.close()

    if verbose:
        print(f"Finished writing to FITS file: {full_path}")

def reorder_hdu(hdul):
    HDUL_order = [
        'SPATIAL_BINS',
        'RADIUS',
        'REDSHIFT', 'REDSHIFT_MASK', 'REDSHIFT_ERROR',
        'NaI_SNR',
        'EW_NAI', 'EW_NAI_MASK', 'EW_NAI_ERROR',
        'EW_NOEM', 'EW_NOEM_MASK', 'EW_NOEM_ERROR'
        'SFRSD', 'SFRSD_MASK', 'SFRSD_ERROR',
        'V_NaI', 'V_NaI_FRAC', 'V_NaI_MASK', 'V_NaI_ERROR',
        'V_MAX_OUT', 'V_MAX_OUT_MASK', 'V_MAX_OUT_ERROR',
        'MCMC_RESULTS', 'MCMC_16TH_PERC', 'MCMC_84TH_PERC',
        'HII',
        'BPT'
    ]
    primary_hdu = hdul[0]
    sorted_hdus = [primary_hdu]
    sorted_hdus += sorted(
        hdul[1:],
        key=lambda hdu: HDUL_order.index(hdu.name)
        if hdu.name in HDUL_order else len(HDUL_order)
    )
    return fits.HDUList(sorted_hdus)
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

    config = configparser.ConfigParser(allow_no_value=True)
    parsing = True
    while parsing:
        try:
            config.read(config_fil)
            parsing = False
        except configparser.Error as e:
            verbose_print(verbose, f"Error parsing file: {e}")
            verbose_print(verbose, f"Cleaning {config_fil}")
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
                    file.write(line + '\n')

    print("Done.")


def threshold_parser(galname, bin_method, require_ew=True):
    thresholds_path = os.path.join(defaults.get_default_path('config'), 'thresholds.yaml')
    if not os.path.isfile(thresholds_path):
        raise ValueError(f"File not found: {thresholds_path}")
    
    with open(thresholds_path, "r") as f:
        thresholds_file = yaml.safe_load(f)
        
    # Always try to get SNR bins
    try:
        snr_edges = thresholds_file['snr_bins'][bin_method]
    except KeyError:
        print(f"Binning method '{bin_method}' not found in SNR bins.")
        return None

    # Convert edges to floats
    snr_edges = [float('inf') if s == 'inf' else float(s) for s in snr_edges]
    sn_lims = list(zip(snr_edges[:-1], snr_edges[1:]))

    # Safely get EW thresholds dictionary (handle None or missing key)
    ew_thresholds = thresholds_file.get('ew_thresholds') or {}

    # Then look for this galaxy
    ew_raw = ew_thresholds.get(galname, None)
    if ew_raw is None:
        if require_ew:
            print(f"No EW thresholds found for galaxy '{galname}'.")
            return None
        else:
            return {
                "sn_lims": sn_lims,
                "ew_lims": None,
            }

    ew_lims = [float('inf') if ew == 'inf' else float(ew) for ew in ew_raw]

    if len(sn_lims) != len(ew_lims):
        raise ValueError(f"Mismatch: {len(sn_lims)} S/N bins but {len(ew_lims)} EW limits.")

    return {
        "sn_lims": sn_lims,
        "ew_lims": ew_lims
    }

def write_thresholds(galname, ew_lims, overwrite=True, verbose=False):
    def sanitize_thresholds(array):
        return [float(x) if np.isfinite(x) else float('inf') for x in np.asarray(array)]
    
    thresholds_path = os.path.join(defaults.get_default_path('config'), 'thresholds.yaml')
    with open(thresholds_path, "r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data.get("ew_thresholds"), dict):
        data["ew_thresholds"] = {}

    # Force clean Python-native types
    ew_lims = sanitize_thresholds(ew_lims)

    if galname not in data["ew_thresholds"] or not data["ew_thresholds"][galname]:
        data["ew_thresholds"][galname] = ew_lims
    else:
        if not overwrite:
            print(f"{galname} already has EW lims of {data['ew_thresholds'][galname]}")
        else:
            verbose_print(verbose, f"Overwriting thresholds for {galname}")
            data["ew_thresholds"][galname] = ew_lims

    with open(thresholds_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def use_sdss_header(galname, bin_method, hduname, sdss_header, desc = None, 
                    ignore_keywords = ['XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'PCOUNT', 'GCOUNT', 'EXTNAME'], remove_keywords = []):
    header = sdss_header
    header = {
        hduname:{
            "DESC":(f"{galname}-{bin_method} {desc}",""),
            "EXTNAME":(hduname, "Extension name"),
        }
    }

    for key, value, comment in zip(sdss_header.keys(), sdss_header.values(), sdss_header.comments):
        if key in ignore_keywords or key in remove_keywords:
            continue

        else:
            header[hduname][key] = (value, comment)



    
    return header