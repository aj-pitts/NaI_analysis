import numpy as np
from astropy.io import fits
import argparse
import warnings
import os
from tqdm import tqdm
from modules import defaults, util, file_handler, plotter


def boxcar_EW(wavelengths, wavelength_error, normflux, normflux_error):
    ones = np.ones(len(normflux))
    dLambda = np.gradient(wavelengths) #np.array([np.median(np.diff(wavelengths)) * len(normflux)]) # Delta lambda
    dLambda_sigma = np.gradient(wavelength_error) #np.array([np.median(np.diff(wavelength_error)) * len(normflux)])

    EW = np.sum(  (( ones - normflux ) * dLambda)  )
    EW_err = np.sqrt( np.sum( ((dLambda * normflux_error)**2 + ((ones - normflux) * dLambda_sigma)**2) ) )
    return EW, EW_err

def measure_EW(galname, bin_method, verbose=False, write_data = True):
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
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    
    ## initialize values
    c = 2.998e5 #speed of light km/s
        
    ## init the data
    cube = fits.open(datapath_dict['LOGCUBE'])
    local = fits.open(datapath_dict['LOCAL'])
    
    flux = cube['FLUX'].data
    ivar = cube['IVAR'].data
    mask = cube['MASK'].data #DAPSPECMASK
    wave = cube['WAVE'].data
    model = cube['MODEL'].data
    model_mask = cube['MODEL_MASK'].data #DAPSPECMASK
    binids = cube['BINID'].data 

    zmap = local['redshift'].data
    zmap_error = local['redshift_error'].data
    zmap_mask = local['redshift_mask'].data


    spatial_bins = binids[0]
    uniqids = np.unique(spatial_bins)
    

    ## define the Na D window bounds
    nad_region = 5885, 5905
    continuum_lims = [(5850, 5870), (5910, 5930)]
    
    ## init empty arrays to write measurements to
    l, ny, nx = flux.shape
    ewmap = np.zeros((ny,nx)) - 999.0
    ewmap_error = np.zeros((ny,nx)) - 999.0
    ewmap_mask = np.zeros((ny,nx))

    ewmap_noem = np.zeros((ny,nx)) - 999.0
    ewmap_noem_error = np.zeros((ny,nx)) - 999.0
    ewmap_noem_mask = np.zeros((ny,nx))

    items = uniqids[1:]
    iterator = tqdm(uniqids[1:], desc="Constructing equivalent width map.") if verbose else items

    ## mask unused spaxels
    w = spatial_bins == -1
    ewmap_mask[w] = 6
    ewmap_noem_mask[w] = 6

    for ID in iterator:
        w = spatial_bins == ID
        ny, nx = np.where(w)
        y, x = ny[0], nx[0]
        
        zbin = zmap[y,x]
        zbin_err = zmap_error[y,x]
        zbin_mask = zmap_mask[y,x]

        if bool(zbin_mask):
            ewmap_mask[w] = zmap_mask[y,x]
            ewmap_noem_mask[w] = zmap_mask[y,x]

        ## shift wavelengths to restframe
        restwave = wave / (1+zbin)
        restwave_sigma = wave * zbin_err / (1 + zbin)**2

        ## extract 1D arrays of the bin
        flux_bin = flux[:, y, x]
        ivar_bin = ivar[:, y, x]
        flux_err = 1 / np.sqrt(ivar_bin)
        mask_bin = util.spec_mask_handler(mask[:, y, x])

        model_bin = model[:, y, x]
        model_mask_bin = util.spec_mask_handler(model_mask[:, y, x])
        
        ## combine DAP masks and finite values
        datamask = np.logical_and(~mask_bin.astype(bool), ~model_mask_bin.astype(bool))
        finite_mask = model_bin > 0
        norm_mask = np.logical_and(datamask, finite_mask)

        ## normalize flux by the model with the masks
        norm_flux = flux_bin[norm_mask] / model_bin[norm_mask]
        norm_error = flux_err[norm_mask] / model_bin[norm_mask]
        wavelength = restwave[norm_mask]
        wavelength_error = restwave_sigma[norm_mask]

        if len(norm_flux) == 0:
            ewmap_mask[w] = 4
            ewmap_noem_mask[w] = 4
            continue

        ## get the indices defining the Na D wavelength region
        nad_inds = np.where((wavelength >= nad_region[0]) & (wavelength <= nad_region[1]))[0]

        if len(nad_inds) < 10:
            ewmap_mask[w] = 4
            ewmap_noem_mask[w] = 4
            continue

        ## extract Na D values
        norm_flux_nad = norm_flux[nad_inds]
        norm_error_nad = norm_error[nad_inds]
        restwave_nad = wavelength[nad_inds]
        restwave_error_nad = wavelength_error[nad_inds]

        ## compute equivalent width
        EW, EW_err = boxcar_EW(restwave_nad, restwave_error_nad, norm_flux_nad, norm_error_nad)

        if np.isfinite(EW):
            ewmap[w] = EW
        else:
            ewmap_mask[w] = 4

        if np.isfinite(EW_err):
            ewmap_error[w] = EW_err
        else:
            ewmap_mask[w] = 5

        ## compute EW again with emline masking
        blue_lims = continuum_lims[0]
        blue_inds = np.where((restwave > blue_lims[0]) & (restwave < blue_lims[1]))[0]

        red_lims = continuum_lims[1]
        red_inds = np.where((restwave > red_lims[0]) & (restwave < red_lims[1]))[0]

        continuum_inds = np.concatenate([blue_inds, red_inds])
        norm_flux_continuum = norm_flux[continuum_inds]

        continuum_filter = (norm_flux_continuum > np.median(norm_flux_continuum) - np.std(norm_flux_continuum)) & (norm_flux_continuum < np.median(norm_flux_continuum) + np.std(norm_flux_continuum))
        continuum_filtered = norm_flux_continuum[continuum_filter]

        median = np.median(continuum_filtered)
        std = np.std(continuum_filtered)
        emline_threshold = median + std

        flux_filter = norm_flux_nad < emline_threshold

        norm_flux_nad = norm_flux_nad[flux_filter]
        norm_error_nad = norm_error_nad[flux_filter]
        restwave_nad = restwave_nad[flux_filter]
        restwave_error_nad = restwave_error_nad[flux_filter]

        if len(norm_flux_nad)<=5:
            ewmap_noem_mask[w] = 4
            continue

        EW_noem, EW_err_noem = boxcar_EW(restwave_nad, restwave_error_nad, norm_flux_nad, norm_error_nad)

        if np.isfinite(EW_noem):
            ewmap_noem[w] = EW_noem
        else:
            ewmap_noem_mask[w] = 4

        if np.isfinite(EW_err_noem):
            ewmap_noem_error[w] = EW_err_noem
        else:
            ewmap_noem_mask[w] = 5

    EW_dict = {"EW Map":ewmap, "EW Map Mask":ewmap_mask, "EW Map Uncertainty":ewmap_error}
    EW_noem_dict = {"EW NOEM Map":ewmap_noem, "EW NOEM Mask":ewmap_noem_mask, "EW NOEM Uncertainty":ewmap_noem_error}

    if write_data:
        ew_mapdict = file_handler.standard_map_dict(galname, EW_dict, HDU_keyword="EW_NAI", IMAGE_units="Angstrom")
        ew_noem_mapdict = file_handler.standard_map_dict(galname, EW_noem_dict, HDU_keyword="EW_NOEM", IMAGE_units="Angstrom")
        file_handler.write_maps_file(galname, bin_method, [ew_mapdict, ew_noem_mapdict], verbose=verbose, preserve_standard_order=True)
    else:
        return EW_dict, EW_noem_dict



def get_args():

    parser = argparse.ArgumentParser(description="Measure equivalent width of ISM Na I absorption throughout an input galaxy and plot the associated maps.")
    
    parser.add_argument('galname', type = str, help = 'Input galaxy name.')
    parser.add_argument('bin_method', type = str, help = 'Input DAP spatial binning method.')

    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)
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

    hdu_name = "EW_NAI"
    units = "Angstrom"
    mapdict = file_handler.standard_map_dict(args.galname, EW_dict, HDU_keyword=hdu_name, IMAGE_units=units)
    
    file_handler.map_file_handler(f"{args.galname}-{args.bin_method}", [mapdict], gal_local_dir, verbose=args.verbose)

    ## create the figures with the map plotter
    plotter.map_plotter(EW_dict['EW Map'], EW_dict['EW Map Mask'], gal_figures_dir, hdu_name, r'$\mathrm{EW_{Na\ D}}$', r'$\left( \mathrm{\AA} \right)$',
                  args.galname, args.bin_method, error=EW_dict['EW Map Uncertainty'],vmin=-0.2,vmax=1.5, s=1)


if __name__ == "__main__":
    args = get_args()
    main(args)