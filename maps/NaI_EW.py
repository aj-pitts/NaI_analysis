import numpy as np
from astropy.io import fits
import argparse
import configparser
import os
from tqdm import tqdm
from ..modules import defaults, util, file_handler, plotter


def measure_EW(cubefil, mapfil, z_guess, verbose=False, bokeh=False):
    """
    Measure the equivalent width (EW) of spectral features in a 3D data cube.

    This function calculates the equivalent width (EW) of spectral features in a
    data cube file and generates corresponding maps. The analysis is based on an
    estimated redshift (`z_guess`) and outputs the results to a specified location.
    Optional visualizations of the results can be generated using Bokeh.

    Parameters
    ----------
    cubefil : str
        Path to the LOGCUBE FITS file from the DAP output.
        
    mapfil : str
        Path to the MAPS FITS file containing additional maps from the DAP.
        
    z_guess : float
        Estimated redshift of the source, used to calculate the sysetmaic rest-frame
        wavelength.
        
    verbose : bool, optional
        If `True`, prints detailed progress messages to the console. Useful for
        debugging or tracking the analysis progress. Default is `False`.
        
    bokeh : bool, optional
        If `True`, generates an interactive visualization of the EW maps using
        Bokeh. Default is `False`.

    Returns
    -------
    results : dict
        A dictionary containing the calculated EW values and any relevant metadata.
        Key contents may include:
        
        - 'EW_map' : 2D numpy.ndarray
            Array of calculated equivalent widths mapped to spatial locations.
            
        - 'error_map' : 2D numpy.ndarray
            Array of uncertainties associated with the equivalent width measurements.
            
        - 'other_results' : dict
            Additional results or parameters, such as fitting statistics or
            derived parameters, if applicable.

    Raises
    ------
    FileNotFoundError
        If either `cubefil` or `mapfil` cannot be found or opened.
        
    ValueError
        If `z_guess` is outside of an expected range or if input data is
        incompatible.

    Notes
    -----
    - This function assumes the spectral data is in a 3D data cube format with
      axes representing spatial dimensions and wavelength.
    - The redshift guess (`z_guess`) is crucial for aligning observed features
      with their rest-frame wavelengths.
    - Requires `astropy` for FITS file handling and optional dependencies if Bokeh
      visualization is enabled.

    Examples
    --------
    Calculate EW and save results to a specific directory with verbose output:

    >>> measure_EW("spectra_cube.fits", "reference_map.fits", 0.03, savepath="results/",
    ...            verbose=True)

    Generate an interactive Bokeh plot:

    >>> measure_EW("spectra_cube.fits", "reference_map.fits", 0.03, bokeh=True)
    
    """
    logger = util.verbose_logger(verbose)

    ## initialize values
    c = 2.998e5 #speed of light km/s
    
    ## check if the redshift from the config file is in string format
    if isinstance(z_guess, str):
        z_guess = float(z_guess)
        
    ## init the data
    cube = fits.open(cubefil)
    maps = fits.open(mapfil)
    
    flux = cube['FLUX'].data
    ivar = cube['IVAR'].data
    mask = cube['MASK'].data.astype(bool)

    wave = cube['WAVE'].data

    model = cube['MODEL'].data
    model_mask = cube['MODEL_MASK'].data
    
    stellarvel = maps['STELLAR_VEL'].data
    stellarvel_mask = maps['STELLAR_VEL_MASK'].data.astype(bool)
    stellarvel_ivar = maps['STELLAR_VEL_IVAR'].data

    binids = maps['BINID'].data ## 0: stacked spectra, 1: stellar-continuum results, 2: empty, 3: emline model results, 4: empty

    spatial_bins = binids[0]
    uniqids = np.unique(spatial_bins)
    

    ## define the Na D window bounds
    region = 5880, 5910
    
    ## init empty arrays to write measurements to
    l, ny, nx = flux.shape
    ewmap, ewmap_unc = np.zeros((ny,nx)) -999.0
    ewmap_qualflag, ewmap_mask = np.ones((ny,nx))
    wavecube = np.zeros(flux.shape)

    for ID in tqdm(uniqids[1:], desc="Constructing equivalent width map."):
        w = spatial_bins == ID
        y, x = np.where(w)

        bin_check = util.check_bin_ID(ID, binids, stellarvel, stellarvel_mask)
        ewmap_qualflag[w] = bin_check

        if bin_check != 1:
            ewmap_mask[w] = 0

        ## get the stellar velocity of the bin
        sv = stellarvel[w][0]
        sv_sigma = 1/np.sqrt(stellarvel_ivar[w][0])

        ## Calculate redshift
        z = (sv * (1+z_guess))/c + z_guess
        z_sigma = (sv_sigma/c) * (1 + z_guess)

        # shift wavelengths to restframe
        restwave = wave / (1+z)
        restwave_sigma = wave * z_sigma / (1 + z)**2

        # TODO
        if bokeh:
            for dy,dx in zip(y,x):
                wavecube[np.arange(len(restwave)),dy,dx] = restwave

        # define wavelength boundaries and store a slice object to slice the datacubes
        NaD_window = (restwave>region[0]) & (restwave<region[1])
        NaD_window_inds = np.where(NaD_window)[0]
        slice_inds = (NaD_window_inds[:, None], y, x)

        # slice wavelength and uncertainty arrays to NaD window
        wave_window = restwave[NaD_window_inds]
        wave_window_sigma = restwave_sigma[NaD_window_inds]

        # slice flux and model datacubes to wavelength window and bin
        # take one array from each bin
        flux_sliced = flux[slice_inds][0]
        ivar_sliced = ivar[slice_inds][0]
        flux_sigma_sliced = 1 / np.sqrt(ivar_sliced)
        mask_sliced = mask[slice_inds][0]

        model_sliced = model[slice_inds][0]
        model_mask_sliced = model_mask[slice_inds][0]


        ## compute equivalent width
        cont = np.ones(len(flux_sliced)) # normalized continuum
        dlam = np.diff(wave_window) # Delta lambda
        dlam_sigma = np.hypot(wave_window_sigma[:-1], wave_window_sigma[1:]) # Delta lambda uncertainty

        W_mask = np.logical_or(mask_sliced.astype(bool), model_mask_sliced.astype(bool)) # combined flux and model masks
        
        W = np.sum(( (cont - flux_sliced / model_sliced ) * dlam)[W_mask]) # Equivalent wdith

        W_sigma = np.sqrt( np.sum( ((dlam * flux_sigma_sliced/model_sliced)**2 + ((cont - flux_sliced/model_sliced) * dlam_sigma)**2)[W_mask] ) ) # EW uncertainty

        if not np.isfinite(W):
            ewmap_qualflag[w] = -4
            continue
        
        if not np.isfinite(W_sigma):
            ewmap_qualflag[w] = -5
            continue
        
        ewmap[w] = W
        ewmap_unc[w] = W_sigma
    
    #TODO
    if bokeh:
        keyword = f"{args.galname}-EW-bokeh"
        #plotter.make_bokeh_map(flux, model, ivar, wavecube, ewmap, spatial_bins, savepath, keyword)

    return {"EW Map":ewmap, "EW Map Mask":ewmap_mask, "EW Map Uncertainty":ewmap_unc, "EW Map Quality Flag":ewmap_qualflag}


def get_args():

    parser = argparse.ArgumentParser(description="Measure equivalent width of ISM Na I absorption throughout an input galaxy and plot the associated maps.")
    
    parser.add_argument('galname', type = str, help = 'Input galaxy name.')
    parser.add_argument('bin_method', type = str, help = 'Input DAP spatial binning method.')

    parser.add_argument('-v','--verbose', type = bool, help = "Print verbose outputs (default: False)", action='store_true', default = False)
    parser.add_argument('-nc','--no_corr', type = bool, help = "Perform analysis on the 'NO-CORR' DAP outputs. (default: False)", action='store_true', default = False)
    parser.add_argument('-ow', '--overwrite', type = bool, help = "Write into any existing map file. Overwrite any equivalent data stored inside. (default: True)", action = 'store_true', default = True)
    parser.add_argument('-np', '--new_plot', type = bool, action = "store_true", help = "Plot the maps to a new pdf rather than overwriting existing (default: False)", default = False)
    parser.add_argument('-z', '--redshift', type=str, help='Use this extension to optionally input a galaxy redshift guess. (default: None)', default=None)

    parser.add_argument('--bokeh', type = bool, help = "Broken (default: False)", default = False)
    
    return parser.parse_args()



def main(args):
    ## initialize the verbose logger
    logger = util.verbose_logger(args.verbose)

    ## acquire all relevant datapaths
    logger.info("Acquiring relevant files")
    datapath_dict = file_handler.init_datapaths(args.galname, args.bin_method)
    
    ## redshift from config file if no input
    redshift = args.redshift
    if redshift is None:
        logger.info("Parsing config file for redshift")
        config_file = datapath_dict['CONFIG']
        config = configparser.ConfigParser()
        parsing = True
        while parsing:
            try:
                config.read(config_file)
                parsing = False
            except configparser.Error as e:
                util.verbose_print(args.verbose, f"Error parsing file: {e}")
                util.verbose_print(args.verbose, f"Cleaning {config_file}")
                util.clean_ini_file(config_file, overwrite=True)

        redshift = config['default']['z']
        util.verbose_print(args.verbose, f"Redshift z = {redshift} found in {config_file}")

    ## default to beta-corr data unless -nc argument
    corr_key = 'BETA-CORR'
    if args.no_corr:
        corr_key = 'NO-CORR'

    ## acquire logcube and maps files from datapath dict, raise error if they were not found
    cubefil = datapath_dict[corr_key]['LOGCUBE']
    mapfil = datapath_dict[corr_key]['MAPS']
    if cubefil is None or mapfil is None:
        raise ValueError(f"LOGCUBE or MAPS file not found for {args.galname}-{args.bin_method}-{corr_key}")

    ## compute EW maps
    EW_dict = measure_EW(cubefil, mapfil, redshift, args.verbose, args.bokeh)

    ## initialize the output paths
    root_dir = defaults.get_default_path('data')
    local_dir = os.path.join(root_dir, 'local_outputs')
    gal_local_dir = os.path.join(local_dir, f"{args.galname}-{args.bin_method}")
    gal_figures_dir = os.path.join(gal_local_dir, "figures")

    util.check_filepath([gal_local_dir, gal_figures_dir], mkdir=True, verbose=args.verbose)

    hdu_name = "EQ_WIDTH_NAI"
    units = "Angstrom"
    mapdict = file_handler.standard_map_dict(args.galname, hdu_name, units, EW_dict)
    file_handler.map_file_handler(f"{args.galname}-{args.bin_method}", mapdict, 
                                  overwrite = args.overwrite, verbose = args.verbose)

    
    ## create the figures with the map plotter
    plotter.map_plotter(EW_dict['EW Map'], EW_dict['EW Map Mask'], hdu_name, r'$\mathrm{EW_{Na\ D}}$', r'$\mathrm{\AA}$',
                  args.galname, args.binmethod, error=EW_dict['EW Map Uncertainty'])


if __name__ == "__main__":
    args = get_args()
    main(args)