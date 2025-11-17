import argparse
import os
from astropy.io import fits

import matplotlib.pyplot as plt

import numpy as np
from modules import util, defaults, file_handler, model_nai

import cmasher as cmr
import warnings

from ifuviewer.interactiveifu import InteractiveIFUViewer



def get_args():    
    parser = argparse.ArgumentParser(description="A script to create an equivalent width map of ISM Na I for beta-corrected DAP outputs.")
    
    parser.add_argument('galname',type=str,help='Input galaxy name.')
    parser.add_argument('bin_method',type=str,help='Input DAP patial binning method.')
    parser.add_argument("--hdu-keys", nargs="+", default=None, help="HDU keywords specifying the maps to be displayed (default: None). If None, all maps will be displayed")
    parser.add_argument('--verbose', help='Print verbose outputs. (Default: False)', action='store_true', default=False)
    
    return parser.parse_args()

def begin_viewing(galname, bin_method, hdu_keys, verbose = False):
    warnings.simplefilter('ignore')
    # mplconfig = os.path.join(defaults.get_default_path('config'), 'figures.mplstyle')
    # plt.style.use(mplconfig)

    util.verbose_print(verbose, f"Preparing interactive viewer for {galname} {bin_method}")
    datapaths = file_handler.init_datapaths(galname, bin_method)

    # setup dictionaries
    mapdict = {}
    maskdict = {}

    # open local maps
    with fits.open(datapaths['LOCAL']) as hdul:
        spatial_bins = hdul['spatial_bins'].data
        redshift = hdul['redshift'].data
        mcmc_cube = hdul['mcmc_results'].data
        mcmc_16 = hdul['mcmc_16th_perc'].data
        mcmc_84 = hdul['mcmc_84th_perc'].data
        vfrac = hdul['v_nai_frac'].data

        # handles which images to grab
        # make a list of each hdul keyword
        if hdu_keys is None:
            hdu_keys = [
                hdu.name
                for hdu in hdul
                if hdu.name != 'PRIMARY'
                and 'ERROR' not in hdu.name # ignore error maps
                and 'MCMC' not in hdu.name # ignore mcmc params (not 2D)
                and 'BINS' not in hdu.name # ignore spatial bins
                and 'BPT' not in hdu.name # ignore for now
                and 'MAX' not in hdu.name # ignore v_out_max
                and 'FRAC' not in hdu.name # ignore v_frac
            ]
        
        util.verbose_print(verbose, f"MAPS to be plot: {hdu_keys}")
        util.verbose_print(verbose, f"Unpacking data...")

        # iterate through hdul keywords and assign to data dicts
        for key in hdu_keys:
            if 'MASK' in key:
                maskkey = key.split('_MASK')[0]
                maskdict[maskkey] = hdul[key].data 
                continue
            mapdict[key] = hdul[key].data
    
    # manually add extra masks
    maskdict['REDSHIFT'] = (redshift<=0.003) # bad redshifts
    maskdict['all'] = (spatial_bins==-1) # unbinned spaxels

    # grab datacubes
    with fits.open(datapaths['LOGCUBE']) as hdul:
        flux = hdul['flux'].data
        model = hdul['model'].data
        wave = hdul['wave'].data

    # construct wavelength datacube
    util.verbose_print(verbose, "Preparing sliced cubes...")
    wave_reshaped = wave[:, np.newaxis, np.newaxis]
    redshift_reshaped = redshift[np.newaxis, :, :]

    wavecube = wave_reshaped / (1 + redshift_reshaped)
    nwave, ny, nx = wavecube.shape

    # slice each wavelength to NaD (better this way than having xlims in IFUviewer, far less data)
    NaD_window = (5875, 5915)
    mask_cube = (wavecube >= NaD_window[0]) & (wavecube <= NaD_window[1])

    nwave_per_pix = np.sum(mask_cube, axis=0)
    nwave_lim = np.max(nwave_per_pix)

    flux_subcube = np.full((nwave_lim, ny, nx), np.nan)
    model_subcube = np.full((nwave_lim, ny, nx), np.nan)
    wave_subcube = np.full((nwave_lim, ny, nx), np.nan)

    for y in range(ny):
        for x in range(nx):
            select = mask_cube[:, y, x]
            flux_subcube[:nwave_per_pix[y,x], y, x] = flux[select, y, x]
            model_subcube[:nwave_per_pix[y,x], y, x] = model[select, y, x]
            wave_subcube[:nwave_per_pix[y,x], y, x] = wavecube[select, y, x]

    util.verbose_print(verbose, "Preparing line-profile model cube...")
    # create the line-profile model datacube
    specmod_subcube = np.full((nwave_lim, ny, nx), np.nan)

    for binid in np.unique(spatial_bins):
        if binid == -1:
            continue
        w = binid == spatial_bins
        ny, nx = np.where(w)
        y, x = ny[0], nx[0]
        theta = mcmc_cube[:, y, x]
        if any(np.array(theta) == 0):
            continue
        z = redshift[y,x]
        wav = wave_subcube[:nwave_per_pix[y,x], y, x]

        moddict = model_nai.model_NaI(theta, z, wav)
        specmod_subcube[:nwave_per_pix[y,x], w] = moddict['modflx'][:, None]
            
    util.verbose_print(verbose, "Preparing imshow kwargs...")
    # setup imshow kwargs
    imshow_kwargs = {}
    for key in mapdict.keys():
        if key.upper() == 'NAI_SNR':
            c = cmr.sapphire
            immin = 0
            immax = 100
        elif key.upper() == 'EW_NAI' or key == 'EW_NOEM':
            c = cmr.gem
            immin = -0.5
            immax = 2
        elif key.upper() == 'SFRSD':
            c = util.seaborn_palette('rainbow')
            immin = -5
            immax = -1
        elif key.upper() == 'V_NAI':
            c = util.seaborn_palette('seismic')
            immin = -200
            immax = 200
        else:
            c = util.seaborn_palette('viridis')
            immin = None
            immax = None

        # extent = [32.4, -32.6, -32.4, 32.6] ### extent does not work
        imshow_kwargs[key] = dict(cmap = c, vmin = immin, vmax = immax, origin='lower')

    # setup imshow plot titles
    imshow_titles = {
        'RADIUS':r'$R/R_e$',
        'REDSHIFT':r'$z$',
        'NaI_SNR':r'$S/N_{\mathrm{Na\ D}}$',
        'EW_NAI':r'$\mathrm{EW_{Na\ D}}$',
        'EW_NOEM':r'$\mathrm{EW_{Na\ D,\ mask}}$',
        'SFRSD':r'$\mathrm{\Sigma_{SFR}}$',
        'E(B-V)':r'$E(B-V)$',
        'V_NaI':r'$v_{\mathrm{cen}}$'
    }

    # setup spaxinfo using MCMC values
    util.verbose_print(verbose, "Preparing spaxinfo cubes...")
    spaxinfo = {
        r'$\lambda_0$':None,
        r'$\mathrm{log}\ N$':None,
        r'$b_D$':None,
        r'$C_f$':None
    }
    for i, key in enumerate(list(spaxinfo.keys())):
        arr = np.zeros((3, spatial_bins.shape[0], spatial_bins.shape[1]))
        arr[0] = mcmc_cube[i, :, :]
        arr[1] = mcmc_16[i, :, :]
        arr[2] = mcmc_84[i, :, :]

        spaxinfo[key] = arr

    # also add vfrac to spaxinfo
    spaxinfo[r'$P(\Delta v)$'] = vfrac

    # add vlines of nad rest and lambda cen
    vlines = {
        'rest':[5891.5833, 5897.5581],
        'lambdacen':mcmc_cube[0, :, :]
    }

    viewer = InteractiveIFUViewer(mapdict, flux_subcube, maps_masks = maskdict, modelcube = specmod_subcube,
                                  continuumcube = model_subcube, binmap = spatial_bins,
                                  wavelength = wave_subcube, vlines = vlines, spax_info = spaxinfo,
                                  map_kwargs = imshow_kwargs, map_titles = imshow_titles)
    viewer.show()





if __name__ == "__main__":
    args = get_args()
    begin_viewing(args.galname, args.bin_method, args.hdu_keys, args.verbose)
