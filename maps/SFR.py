import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, join
from glob import glob
import os
import argparse
import warnings
from tqdm import tqdm
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.coordinates import Angle

from modules import defaults, util, file_handler, plotter

## TODO: add detailed docstrings and comments

## TODO: propagate emission line flux uncertainty through correct dust
def correct_dust(F_Ha, F_Hb, HaHb_ratio = 2.87, Rv = 3.1, k_Ha = 2.45, k_Hb = 3.65):
    """
    Corrects H-alpha flux for dust attenuation using the Balmer decrement method.

    Parameters
    ----------
    F_Ha : array_like
        Observed H-alpha flux values.
    F_Hb : array_like
        Observed H-beta flux values.
    HaHb_ratio : float, optional
        Theoretical H-alpha/H-beta flux ratio for case B recombination in an 
        ionized gas. The default is 2.87, which assumes a typical electron density 
        and temperature for HII regions.
    Rv : float, optional
        Total-to-selective extinction ratio, typically 3.1 for the Milky Way. 
        (Note: This parameter is currently not used in the calculation.)
    k_Ha : float, optional
        Extinction coefficient at the wavelength of H-alpha (6563 Å).
    k_Hb : float, optional
        Extinction coefficient at the wavelength of H-beta (4861 Å).

    Returns
    -------
    F_corr : ndarray
        Dust-corrected H-alpha flux values. Returns `NaN` for entries where the 
        input fluxes are non-positive or non-finite.

    Notes
    -----
    This function corrects the observed H-alpha flux for dust attenuation based on 
    the observed H-alpha to H-beta flux ratio and a theoretical ratio (`HaHb_ratio`).
    It uses the Cardelli, Clayton, and Mathis (1989) extinction law by default then converts 
    to a gas attenuation factor (A_gas) at the H-alpha wavelength to derive the corrected flux.

    Examples
    --------
    >>> import numpy as np
    >>> F_Ha = np.array([100, 50, 0, -20])
    >>> F_Hb = np.array([30, 20, 10, 0])
    >>> correct_dust(F_Ha, F_Hb)
    array([ corrected_flux_value_1, corrected_flux_value_2, nan, nan ])
    """
    E_BV = np.zeros(F_Ha.shape)
    F_corr = np.zeros(F_Ha.shape)

    w = (F_Ha > 0) & (np.isfinite(F_Ha)) & (F_Hb > 0) & (np.isfinite(F_Hb))

    E_BV[w] = (2.5 / (k_Hb - k_Ha)) * np.log10((F_Ha[w]/F_Hb[w])/HaHb_ratio)
    A_gas = E_BV * k_Ha

    power = 0.4 * A_gas
    F_corr[w] = F_Ha[w] * (10 ** power[w])
    F_corr[~w] = np.nan

    return {'H_alpha':F_corr, 'E_BV':E_BV}


def compute_SFR(flux_ha, bin_width, stellar_velocity, redshift, H0 = 70 * u.km / u.s / u.Mpc, c = 2.998e5 * u.km / u.s):
    """
    Computes the star formation rate (SFR) and the star formation rate surface density (SFRSD)
    given H-alpha flux, stellar velocity, and redshift, with optional uncertainty propagation.

    Parameters
    ----------
    flux_ha : float or tuple
        H-alpha flux in units of erg/s/cm^2. If a tuple, the first value is the flux
        and the second value is the uncertainty.

    stellar_velocity : float or tuple
        Stellar velocity in km/s. If a tuple, the first value is the velocity and
        the second value is the uncertainty.

    redshift : float
        Redshift of the source galaxy.

    H0 : `~astropy.units.Quantity`, optional
        Hubble constant, by default set to 70 km/s/Mpc.

    c : `~astropy.units.Quantity`, optional
        Speed of light, by default set to 2.998e5 km/s.

    Returns
    -------
    dict
        Dictionary containing the computed star formation properties:
        - 'SFR' : float
            The logarithmic star formation rate in units of solar masses per year.
        - 'SFR Uncertainty' : float or None
            Uncertainty in the SFR if input uncertainties are provided; otherwise, None.
        - 'SFRSD' : float
            The logarithmic star formation rate surface density in units of solar masses per year per kpc^2.
        - 'SFRSD Uncertainty' : float or None
            Uncertainty in the SFRSD if input uncertainties are provided; otherwise, None.

    Notes
    -----
    The SFR is calculated using the luminosity of H-alpha emission:
    
        SFR = log10(L / 10^41.27)
    
    where L is the luminosity computed as:
    
        L = 4 * pi * D^2 * flux_ha * 1e-17
    
    with D representing the luminosity distance derived from redshift and stellar velocity.
    SFRSD is derived by normalizing SFR by the area of the spaxel, with angular size 0.2 arcsec.

    If uncertainties are provided for both H-alpha flux and stellar velocity, they are
    propagated to compute uncertainties for both SFR and SFRSD.

    """
    # get the local systematic redshift
    z = ((stellar_velocity * (1 + redshift))/c.value + redshift)

    # compute the physical separation (length) of the spaxel
    D = c * z / H0 # physical distance
    theta = Angle(0.2, unit='arcsec') # angular size of 1 spaxel
    theta *= bin_width # angular size of bin
    sep = (D * theta.radian).to(u.kpc).value # physical separation
    
    # compute the luminosity
    luminosity = 4 * np.pi * (D.to(u.cm).value)**2 * flux_ha * 1e-17

    # compute the SFR and SFR surface density
    SFR = np.log10(luminosity) - 41.27
    SFRSD = np.log10( (10**SFR) / (sep**2) )
    
    ## TODO: ERRORS
    # if uncertainties were input, propagate them
    # if flux_ha_sigma is not None and sv_sigma is not None:
    #     z_sigma = (sv_sigma/c) * (1 + redshift)
    #     D_sigma = (c * z_sigma / H0)
    #     sep_sigma = (D_sigma * theta.radian).to(u.kpc).value
    #     luminosity_sigma = np.sqrt( (8 * np.pi * (D.to(u.cm).value) * flux_ha * 1e-17 * (D_sigma.to(u.cm).value))**2 + (4 * np.pi * (D.to(u.cm).value)**2 * 1e-17 * flux_ha_sigma)**2 )
    #     SFR_sigma = luminosity_sigma / luminosity / np.log(10)
    #     SFRSD_sigma = (1 / ( (10**SFR / sep**2) * np.log(10))) * np.sqrt( ( (10**SFR * np.log(10) * SFR_sigma) / sep**2 )**2 + ( (2 * 10**SFR * sep_sigma) / sep**3)**2 )
    # else:
    #     SFR_sigma = None
    #     SFRSD_sigma = None

    return SFRSD

def SFR_map(galname, bin_method, flux_key = "GFLUX", verbose = False, write_data = True):
    """
    Measure the star formation rate (SFR) and star formation rate surface density (SFRSD) of 
    spectral features in a 3D data cube.

    This function calculates the SFR and SFRSD using spectral features in a
    data cube file and generates corresponding maps. The analysis is based on an
    estimated redshift (`redshift`) and outputs the results to a specified location.
    Optional visualizations of the results can be generated using Bokeh.

    Parameters
    ----------
    map_fil : str
        Path to the MAPS FITS file containing additional maps from the DAP.
        
    redshift : float
        Estimated redshift of the source, used to calculate the sysetmaic rest-frame
        wavelength.

    flux_key : str, optional
        String used to slice the DAP MAPS EMLINE data by the field `f"EMLINE_{flux_key}"`. Default 
        and preferred is "GFLUX". See the [MaNGA DAP DATAMODEL](https://sdss-mangadap.readthedocs.io/en/latest/datamodel.html) 
        for additional information.
        
    verbose : bool, optional
        If `True`, prints detailed progress messages to the console. Useful for
        debugging or tracking the analysis progress. Default is `False`.
        
    bokeh : bool, optional
        If `True`, generates an interactive visualization of the maps using
        Bokeh. Default is `False`.

    Returns
    -------
    results : dict
        A dictionary containing the calculated SFR and SFRSD values and their 
        associated metadata.
        Key contents include:

        - 'SFRSD Map' : 2D numpy.ndarray
            Array of calculated SFRSDs mapped to spatial locations.
            
        - 'SFRSD Mask' : 2D numpy.ndarray
            Array of boolean integers; data quality mask for the SFRSD measurements.
            
        - 'SFRSD Uncertainty' : dict
            Array of propagated uncertainties associated with the SFRSD measurements.
    """
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    redshift = datapath_dict['Z']
    # open the MAPS file
    with fits.open(datapath_dict['MAPS']) as maps:
        # stellar kinematics
        stellar_vel = maps['STELLAR_VEL'].data
        stellar_vel_ivar = maps['STELLAR_VEL_IVAR'].data
        stellar_vel_mask = maps['STELLAR_VEL_MASK'].data

        # bin ids
        binids = maps['BINID'].data
        spatial_bins = binids[0]
        uniqids = np.unique(spatial_bins)
        
        emline_key = f"EMLINE_{flux_key}"
        
        # init the data
        flux = maps[emline_key].data
        ivar = maps[f"{emline_key}_IVAR"].data
        mask = maps[f"{emline_key}_MASK"].data

    # init empty maps
    SFRSD_map = np.zeros(spatial_bins.shape) - 999.0
    SFRSD_sigma = np.zeros(spatial_bins.shape) - 999.0
    SFRSD_mask = np.zeros(spatial_bins.shape)


    # slice values for H alpha and H beta
    ha = flux[23]
    ha_err = 1/np.sqrt(ivar[23])
    ha_mask = mask[23]

    hb = flux[14]
    hb_err = 1/np.sqrt(ivar[14])
    hb_mask = mask[14]

    # correct the entire flux map
    dust_corrections = correct_dust(ha, hb)
    ha_flux_corr = dust_corrections['H_alpha']
    extinction_map = dust_corrections['E_BV']
    
    ## mask unused spaxels
    w = spatial_bins == -1
    SFRSD_mask[w] = 6

    items = uniqids[1:]
    iterator = tqdm(uniqids[1:], desc=f"Constructing SFR map from {flux_key}") if verbose else items
    # loop through each bin and compute SFR

    for ID in iterator:
        w = spatial_bins == ID
        y, x = np.where(w)

        bin_check = util.check_bin_ID(ID, spatial_bins, DAPPIXMASK_list=[stellar_vel_mask], 
                                      stellar_velocity_map=stellar_vel)
        SFRSD_mask[w] = bin_check

        # compute SFR
        ha_flux = ha_flux_corr[y[0], x[0]]
        sv = stellar_vel[y[0], x[0]]

        bin_w = np.sqrt(np.sum(w)) # bin_width for square bins
        sfrsd = compute_SFR(ha_flux, bin_w, sv, redshift)

        if not np.isfinite(sfrsd):
            SFRSD_mask[w] = 5
            continue
        SFRSD_map[w] = sfrsd

    SFR_dict = {"SFRSD Map":SFRSD_map, "SFRSD Mask":SFRSD_mask, "SFRSD Uncertainty":SFRSD_sigma}
    if write_data:
        sfr_mapdict = file_handler.standard_map_dict(galname, SFR_dict, HDU_keyword="SFRSD", IMAGE_units=r"$\left( \mathrm{M_{\odot}\ kpc^{-2}\ yr^{-1}\ spaxel^{-1}} \right)$")
        file_handler.write_maps_file(galname, bin_method, [sfr_mapdict], verbose=verbose, preserve_standard_order=True)

        extinction_dict = {"E_BV":extinction_map}
        name = "E(B-V)"
        extinction_header = {
            name:{
                "DESC":(f"{galname} dust extinction",""),
                "HDUCLASS":("MAP", "Data format"),
                "UNITS":("", "Unit of pixel values"),
                "EXTNAME":(name, "Extension name"),
                "AUTHOR":("Andrew Pitts","")
            }
        }
        extinction_mapdict = file_handler.standard_map_dict(galname, extinction_dict, custom_header_dict=extinction_header)
        file_handler.write_maps_file(galname, bin_method, [extinction_mapdict], verbose=verbose)
    else:
        return SFR_dict


def get_args():
    parser = argparse.ArgumentParser(description="A script to create a SFR map from the DAP emline results of a beta-corrected galaxy.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")

    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)
    parser.add_argument('--sflux', action = 'store_true', help = "Calculate SFR using the EMLINE_SFLUX DAP measurements instead of EMLINE_GFLUX. (default: False)", default=False)

    parser.add_argument('--bokeh', help = "Broken (default: False)", default = False)

    return parser.parse_args()


def main(args):
    warnings.filterwarnings("ignore")
    ## set initial arguments
    # verbose logger

    analysisplan = defaults.analysis_plans()

    # set the data correlation key
    corr_key = 'BETA-CORR'

    # initialize directories and paths
    local_data_dir = defaults.get_data_path(subdir='local')
    local_dir = os.path.join(local_data_dir, 'local_outputs')
    gal_local_dir = os.path.join(local_dir, f"{args.galname}-{args.bin_method}", corr_key, analysisplan)
    gal_figures_dir = os.path.join(local_dir, f"{args.galname}-{args.bin_method}", "figures")

    util.check_filepath([gal_local_dir, gal_figures_dir], mkdir=True, verbose=args.verbose)


    # acquire datafiles from file_handler
    filepath_dict = file_handler.init_datapaths(args.galname, args.bin_method, verbose=args.verbose)

    # set the matplotlib style config

    # acquire the redshift
    configuration = file_handler.parse_config(filepath_dict['CONFIG'], verbose=args.verbose)
    redshift = configuration['z']
    util.verbose_print(args.verbose, f"Redshift z = {redshift} found in {filepath_dict['CONFIG']}")
    
    # set the emline flux key
    flux_key = "GFLUX"
    if args.sflux:
        flux_key = "SFLUX"

    cubefil = filepath_dict['LOGCUBE']
    mapfil = filepath_dict['MAPS']
    if cubefil is None or mapfil is None:
        raise ValueError(f"LOGCUBE or MAPS file not found for {args.galname}-{args.bin_method}-{corr_key}")
    
    ## compute SFRs
    sfrdict = SFR_map(mapfil, redshift, flux_key=flux_key, verbose=args.verbose)

    ## prepare write outs
    # strings for FITS file and plots
    hdu_keyword = "SFRSD"
    label = r"$\mathrm{log \Sigma_{SFR}}$"
    unit = r"$\left( \mathrm{M_{\odot}\ kpc^{-2}\ yr^{-1}\ spaxel^{-1}} \right)$"

    mapsdict = file_handler.standard_map_dict(f"{args.galname}", sfrdict, HDU_keyword=hdu_keyword, IMAGE_units=unit)
    file_handler.map_file_handler(f"{args.galname}-{args.bin_method}", [mapsdict], gal_local_dir,
                                  verbose=args.verbose)
    
    plotter.map_plotter(sfrdict['SFRSD Map'], sfrdict['SFRSD Mask'], gal_figures_dir, hdu_keyword, label, unit,
                        args.galname, args.bin_method, verbose=args.verbose, vmin=-2.5, vmax=0, cmap='rainbow')


if __name__ == "__main__":
    args = get_args()
    main(args)