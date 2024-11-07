import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, join
from glob import glob
import os
import argparse
from tqdm import tqdm
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.coordinates import Angle

from modules import util, plotter, file_handler, defaults

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
    It uses the Cardelli, Clayton, and Mathis (1989) extinction law by default, 
    where the dust attenuation in magnitudes is calculated using:

    .. math::
        E(B-V) = \frac{2.5}{k_{H\beta} - k_{H\alpha}} \log_{10} \left( \frac{F_{H\alpha} / F_{H\beta}}{\text{HaHb\_ratio}} \right)

    This is then converted to a gas attenuation factor (A_gas) at the H-alpha 
    wavelength to derive the corrected flux.

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

    return F_corr



def SFR_map(map_fil, redshift, sflux = False, verbose = False, bokeh = False):
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

    sflux : bool, optional
        If `True`, uses the simple summed emission line fluxes (`EMLINE_SFLUX`) for calculating 
        the SFR. By default the Gaussian integrated flux (`EMLINE_GFLUX`) is preferred. See the 
        [MaNGA DAP DATAMODEL](https://sdss-mangadap.readthedocs.io/en/latest/datamodel.html)for additional information.
        
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
        
        - 'SFR Map' : 2D numpy.ndarray
            Array of calculated SFRs mapped to spatial locations.
            
        - 'SFR Mask' : 2D numpy.ndarray
            Array of boolean integers; data quality mask for the SFR measurements.
            
        - 'SFR Uncertainty' : dict
            Array of propagated uncertainties associated with the SFR measurements.

        - 'SFRSD Map' : 2D numpy.ndarray
            Array of calculated SFRSDs mapped to spatial locations.
            
        - 'SFRSD Mask' : 2D numpy.ndarray
            Array of boolean integers; data quality mask for the SFRSD measurements.
            
        - 'SFRSD Uncertainty' : dict
            Array of propagated uncertainties associated with the SFRSD measurements.
    """

    if isinstance(redshift, str):
        redshift = float(redshift)

    # open the MAPS file
    maps = fits.open(map_fil)

    # stellar kinematics
    stellar_vel = maps['STELLAR_VEL'].data
    stellar_vel_ivar = maps['STELLAR_VEL_IVAR'].data
    stellar_vel_mask = maps['STELLAR_VEL_MASK'].data

    # bin ids
    binids = maps['BINID'].data
    spatial_bins = binids[0]
    uniqids = np.unique(spatial_bins)

    # constants
    H0 = 70 * u.km / u.s / u.Mpc
    c = 2.998e5 * u.km / u.s

    # emission line key
    flux_key = 'EMLINE_GFLUX'
    if not gflux:
        flux_key = 'EMLINE_SFLUX'
    
    # initialize dictionary for storing all maps
    datadict = {}

    # init empty maps
    sfrmap, sfrmap_mask, sfrmap_sigma, sfrdensitymap, sfrdensitymap_mask, sfrdensitymap_sigma = np.zeros(spatial_bins.shape)


    # init the data
    flux = maps[flux_key].data
    ivar = maps[f"{flux_key}_IVAR"].data
    mask = maps[f"{flux_key}_MASK"].data

    # slice values for H alpha and H beta
    ha = flux[23]
    ha_err = 1/np.sqrt(ivar[23])
    ha_mask = mask[23]

    hb = flux[14]
    hb_err = 1/np.sqrt(ivar[14])
    hb_mask = mask[14]

    # correct the entire flux map
    ha_flux_corr = correct_dust(ha, hb)
    
    # loop through each bin and compute SFR
    for ID in tqdm(uniqids[1:], desc=f"Constructing SFR map from {flux_key}"):
        mask_SFR = False

        w = spatial_bins == ID
        y, x = np.where(w)

        if not util.check_bin_ID(ID, binids, stellar_velocity_map = stellar_vel, stellar_vel_mask = stellar_vel_mask):
            mask_SFR = True

        ha_med = np.median(ha_flux_corr[w])
        if ha_med <= 0:
            mask_SFR = True

        sv = (np.median(stellar_vel[w]), np.median(1/np.sqrt(stellar_vel_ivar[w])))
        z = (((sv[0] * (1+redshift))/c.value + redshift), (sv[1]/c) * (1 + redshift))
        D = ((c * z[0] / H0), (c * z[1] / H0))
        theta = Angle(0.2, unit='arcsec')
        s = ((D[0] * theta.radian).to(u.kpc).value, (D[1] * theta.radian).to(u.kpc).value)
        
        luminosity = 4 * np.pi * (D[0].to(u.cm).value)**2 * ha_med * 1e-17
        luminosity_err = np.sqrt( (8 * np.pi * (D[0].to(u.cm).value) * ha_med[0] * 1e-17 * (D[1].to(u.cm).value))**2 + (4 * np.pi * (D[0].to(u.cm).value)**2 * 1e-17 * ha_med[1])**2 )
        L = (luminosity, luminosity_err)

        SFR = (np.log10(L[0]) - 41.27, L[1]/ L / np.log(10))
        sfrmap[w] = SFR[0]
        sfrmap_sigma[w] = SFR[1] 

        SFR_surface_density = np.log10( (10**SFR[0]) / (s[0]**2) )
        SFR_surface_density_err = (1 / ( (10**SFR[0] / s[0]**2) * np.log(10))) * np.sqrt( ( (10**SFR[0] * np.log(10) * SFR[1]) / s[0]**2 )**2 + ( (2 * 10**SFR[0] * s[1]) / s[0]**3)**2 )
        sigma_SFR = (SFR_surface_density, SFR_surface_density_err)
        sfrdensitymap[w] = sigma_SFR[0]
        sfrdensitymap_sigma[w] = sigma_SFR[1]

        for value in [SFR[0], SFR[1], sigma_SFR[0], sigma_SFR[1]]:
            if not np.isfinite(value):
                mask_SFR = True

        if not mask_SFR:
            sfrmap_mask[w] = 1
            sfrdensitymap_mask[w] = 1
            
    datadict = {'SFR Map':sfrmap, 'SFR Mask':sfrmap_mask, 'SFR Uncertainty':sfrmap_sigma, 
                'SFRSD Map':sfrdensitymap, 'SFRSD Mask':sfrdensitymap_mask, 'SFRSD Uncertainty':sfrdensitymap_sigma}
        
    return datadict


def get_args():
    parser = argparse.ArgumentParser(description="A script to create a SFR map from the DAP emline results of a beta-corrected galaxy.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")

    parser.add_argument('-v','--verbose', type = bool, help = "Print verbose outputs (default: False)", action='store_true', default = False)
    parser.add_argument('-nc','--no_corr', type = bool, help = "Perform analysis on the 'NO-CORR' DAP outputs. (default: False)", action='store_true', default = False)
    parser.add_argument('--sflux', type = bool, action = 'store_true', help = "Calculate SFR using the EMLINE_SFLUX DAP measurements instead of EMLINE_GFLUX. (default: False)", default=False)
    parser.add_argument('-ow', '--overwrite', type = bool, help = "Write into any existing map file. Overwrite any equivalent data stored inside. (default: True)", action = 'store_true', default = True)
    parser.add_argument('-np', '--new_plot', type = bool, action = "store_true", help = "Plot the maps to a new pdf rather than overwriting existing (default: False)", default = False)
    parser.add_argument('-z', '--redshift', type=str, help='Use this extension to optionally input a galaxy redshift guess. (default: None)', default=None)

    parser.add_argument('--bokeh', type = bool, help = "Broken (default: False)", default = False)

    return parser.parse_args()

def main(args):
    logger = util.verbose_logger(verbose=args.verbose)
    logger.info("Intitalizing directories and paths.")

    # intialize directories and paths
    filepath_dict = file_handler.init_datapaths(args.galname, args.bin_method, verbose=args.verbose)
    plt.style.use(defaults.matplotlib_rc())
    redshift = args.redshift
    
    if redshift is None:
        redshift = util.redshift_from_config(filepath_dict['CONFIG'], verbose=args.verbose)
        util.verbose_print(args.verbose, f"Redshift z = {redshift} found in {filepath_dict['CONFIG']}")
    

    corr_key = 'BETA-CORR'
    if args.no_corr:
        corr_key = 'NO-CORR'

    sfrdict = SFR_map(filepath_dict[corr_key]['MAPS'], redshift, sflux = args.sflux, verbose=args.verbose, bokeh=args.bokeh)

    
    hdu_keyword = "SFR_HA_GFLUX"
    if args.sflux:
        hdu_keyword = "SFR_HA_SFLUX"

    label = r"$\mathrm{log\ SFR_{H\alpha}"
    units = r"(M_{\odot}\ yr^{-1}\ spaxel^{-1})}$"

    mapsdict = file_handler.standard_map_dict(f"{args.galname}", hdu_keyword, units, sfrdict)
    file_handler.map_file_handler(f"{args.galname}-{args.bin_method}", )


if __name__ == "__main__":
    args = get_args()
    if args.imgftype == "pdf" or args.imgftype == "png":
        pass
    else:
        raise ValueError(f"{args.imgftype} not a valid value for the output image filetype.\nAccepted formats [pdf/png]\nDefault: pdf")
    main(args)