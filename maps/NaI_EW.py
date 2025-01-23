import numpy as np
from astropy.io import fits
import argparse
import warnings
import os
from tqdm import tqdm
from modules import defaults, util, file_handler, plotter

def measure_EW(cubefil, mapfil, z_guess, verbose=False, bokeh=False):
    """
    Measure the equivalent width (EW) of spectral features in a DAP 3D data cube.

    This function calculates the equivalent width (EW) of the neutral sodium absorbtion
    doublet in a data cube file and generates corresponding maps. The analysis is based on an
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
        Key contents include:
        
        - 'EW Map' : 2D numpy.ndarray
            Array of calculated equivalent widths mapped to spatial locations.

        - 'EW Map Mask' : 2D numpy.ndarray
            Array of boolean integers; data quality mask for the EW measurements. Values of 1
            (True) are poor quality.
            
        - 'EW Map Uncertainty' : 2D numpy.ndarray
            Array of uncertainties associated with the equivalent width measurements.

    Examples
    --------
    Calculate EW and save results to a specific directory with verbose output:

    >>> measure_EW("spectra_cube.fits", "reference_map.fits", 0.03,
    ...            verbose=True)

    Generate an interactive Bokeh plot:

    >>> measure_EW("spectra_cube.fits", "reference_map.fits", 0.03, bokeh=True)
    
    """

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
    mask = cube['MASK'].data #DAPSPECMASK

    wave = cube['WAVE'].data

    model = cube['MODEL'].data
    model_mask = cube['MODEL_MASK'].data #DAPSPECMASK
    
    stellarvel = maps['STELLAR_VEL'].data
    stellarvel_mask = maps['STELLAR_VEL_MASK'].data #DAPPIXMASK
    stellarvel_ivar = maps['STELLAR_VEL_IVAR'].data

    binids = maps['BINID'].data 
    spatial_bins = binids[0]

    uniqids = np.unique(spatial_bins)
    

    ## define the Na D window bounds
    region = 5880, 5910
    
    ## init empty arrays to write measurements to
    l, ny, nx = flux.shape
    ewmap = np.zeros((ny,nx)) - 999.0
    ewmap_unc = np.zeros((ny,nx)) - 999.0
    ewmap_mask = np.zeros((ny,nx))
    wavecube = np.zeros(flux.shape)

    items = uniqids[1:]
    iterator = tqdm(uniqids[1:], desc="Constructing equivalent width map.") if verbose else items


    ## mask unused spaxels
    w = spatial_bins == -1
    ewmap_mask[w] = 6

    for ID in iterator:
        w = spatial_bins == ID
        y, x = np.where(w)

        bin_check = util.check_bin_ID(ID, spatial_bins, DAPPIXMASK_list=[stellarvel_mask],
                                      stellar_velocity_map=stellarvel)
        
        ewmap_mask[w] = bin_check

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
        NaD_window_inds_wavelength = np.insert(NaD_window_inds, 0, int(NaD_window_inds[0]-1))
        slice_inds = (NaD_window_inds[:, None], y[0], x[0])

        # slice wavelength and uncertainty arrays to NaD window
        wave_window = restwave[NaD_window_inds_wavelength]
        wave_window_sigma = restwave_sigma[NaD_window_inds_wavelength]

        # slice flux and model datacubes to bin, spaxel, and wavelength window
        flux_sliced = np.array(flux[slice_inds]).ravel()
        ivar_sliced = np.array(ivar[slice_inds]).ravel()
        flux_sigma_sliced = 1 / np.sqrt(ivar_sliced)

        mask_sliced = np.array(mask[slice_inds]).ravel()
        model_sliced = np.array(model[slice_inds]).ravel()
        model_mask_sliced = np.array(model_mask[slice_inds]).ravel()

        ## change the DAPSPECMASKS from bitmasks to boolean array masks
        flux_mask_bool = util.spec_mask_handler(mask_sliced).astype(bool)
        model_mask_bool = util.spec_mask_handler(model_mask_sliced).astype(bool)
        W_mask = np.logical_or(flux_mask_bool, model_mask_bool) # combined flux and model masks

        ## compute equivalent width
        cont = np.ones(len(flux_sliced)) # normalized continuum
        dlam = np.diff(wave_window)# Delta lambda
        dlam_sigma = np.hypot(wave_window_sigma[:-1], wave_window_sigma[1:]) # Delta lambda uncertainty


        W = np.sum(( (cont - flux_sliced / model_sliced ) * dlam)[~W_mask]) # Equivalent width, masked before summming

        W_sigma = np.sqrt( np.sum( ((dlam * flux_sigma_sliced/model_sliced)**2 + ((cont - flux_sliced/model_sliced) * dlam_sigma)**2)[~W_mask] ) ) # EW uncertainty
        

        if not np.isfinite(W) or not np.isfinite(W_sigma):
            if not np.isfinite(W):
                ewmap[w] = -999
                ewmap_mask[w] = 4
            
            if not np.isfinite(W_sigma):
                ewmap_unc[w] = -999
                ewmap_mask[w] = 5

            continue

        ewmap[w] = W
        ewmap_unc[w] = W_sigma
    
    #TODO
    if bokeh:
        keyword = f"{args.galname}-EW-bokeh"
        #plotter.make_bokeh_map(flux, model, ivar, wavecube, ewmap, spatial_bins, savepath, keyword)

    return {"EW Map":ewmap, "EW Map Mask":ewmap_mask, "EW Map Uncertainty":ewmap_unc}


def get_args():

    parser = argparse.ArgumentParser(description="Measure equivalent width of ISM Na I absorption throughout an input galaxy and plot the associated maps.")
    
    parser.add_argument('galname', type = str, help = 'Input galaxy name.')
    parser.add_argument('bin_method', type = str, help = 'Input DAP spatial binning method.')

    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)
    parser.add_argument('-ow', '--overwrite', help = "Write into any existing map file. Overwrite any equivalent data stored inside. (default: True)", action = 'store_false', default = True)
    parser.add_argument('-z', '--redshift', type=str, help='Use this extension to optionally input a galaxy redshift guess. (default: None)', default=None)

    parser.add_argument('--bokeh', type = bool, help = "Broken (default: False)", default = False)
    
    return parser.parse_args()



def main(args):
    # Suppress all warnings
    warnings.filterwarnings("ignore")


    analysisplan = defaults.analysis_plans()

    ## acquire all relevant datapaths
    datapath_dict = file_handler.init_datapaths(args.galname, args.bin_method)
    
    ## redshift from config file if no input
    redshift = args.redshift
    if redshift is None:
        configuration = file_handler.parse_config(datapath_dict['CONFIG'], verbose=args.verbose)
        redshift = configuration['z']
        util.verbose_print(args.verbose, f"Redshift z = {redshift} found in {datapath_dict['CONFIG']}")

    ## default to beta-corr data unless -nc argument
    corr_key = 'BETA-CORR'

    ## acquire logcube and maps files from datapath dict, raise error if they were not found
    cubefil = datapath_dict['LOGCUBE']
    mapfil = datapath_dict['MAPS']
    if cubefil is None or mapfil is None:
        raise ValueError(f"LOGCUBE or MAPS file not found for {args.galname}-{args.bin_method}-{corr_key}")

    ## compute EW maps
    EW_dict = measure_EW(cubefil, mapfil, redshift, verbose = args.verbose, bokeh = args.bokeh)

    ## initialize the output paths
    local_data_dir = defaults.get_data_path('local')
    local_dir = os.path.join(local_data_dir, 'local_outputs')
    gal_local_dir = os.path.join(local_dir, f"{args.galname}-{args.bin_method}", corr_key, analysisplan)
    gal_figures_dir = os.path.join(local_dir, f"{args.galname}-{args.bin_method}", "figures")

    util.check_filepath([gal_local_dir, gal_figures_dir], mkdir=True, verbose=args.verbose)

    hdu_name = "EQ_WIDTH_NAI"
    units = "Angstrom"
    mapdict = file_handler.standard_map_dict(args.galname, hdu_name, units, EW_dict)
    
    # file_handler.simple_file_handler(f"{args.galname}-{args.bin_method}", mapdict, 'EW-map', gal_local_dir,
    #                                  overwrite= args.overwrite, verbose= args.verbose)
    
    file_handler.map_file_handler(f"{args.galname}-{args.bin_method}", mapdict, gal_local_dir, verbose=args.verbose)

    ## create the figures with the map plotter
    plotter.map_plotter(EW_dict['EW Map'], EW_dict['EW Map Mask'], gal_figures_dir, hdu_name, r'$\mathrm{EW_{Na\ D}}$', r'$\left( \mathrm{\AA} \right)$',
                  args.galname, args.bin_method, error=EW_dict['EW Map Uncertainty'],vmin=-0.2,vmax=1.5, s=1)


if __name__ == "__main__":
    args = get_args()
    main(args)