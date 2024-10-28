import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse
import configparser
import os
from glob import glob
from tqdm import tqdm
import logging
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from modules.util import clean_ini_file, check_filepath
from modules.interactive_plot import make_bokeh_map


def make_EW_map(cubefil,mapfil,z_guess,savepath,vmin=-0.2,verbose=False,bokeh=False):

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
    stellar_model_results = binids[1]
    emline_model_results = binids[3]
    
    uniqids = np.unique(spatial_bins[0])
    

    ## define the Na D window bounds
    region = 5880, 5910
    
    ## init empty arrays to write measurements to
    l, ny, nx = flux.shape
    ewmap, ewmap_unc, ewmap_mask = np.zeros((ny,nx))
    wavecube = np.zeros(flux.shape)

    badbins = {
        'Model Results':[],
        'Stellar Vel Mask':[],
        'Stellar Vel':[],
        'Infinite EW':[],
        'Infinite EW_err':[]
    }

    for ID in tqdm(uniqids[1:], desc="Constructing equivalent width map."):
        mask_EW = False
        w = spatial_bins == ID
        y, x = np.where(w)

        ## if bad stellar model or emline fits, skip the spaxel
        if stellar_model_results[w][0] < 0 or emline_model_results[w][0] < 0:
            badbins['Model Results'].append(ID)
            mask_EW = True
            
        ## if stellar kinematics are masked, skip the spaxel
        if not stellarvel_mask[w][0]:
            badbins['Stellar Vel Mask'].append(ID)
            mask_EW = True

        ## get the stellar velocity of the bin
        sv = stellarvel[w][0]
        sv_sigma = 1/np.sqrt(stellarvel_ivar[w][0])

        if abs(sv) > 5 * np.std(stellarvel) + np.median(stellarvel):
            badbins['Stellar Vel'].append(ID)
            mask_EW = True

        ## Calculate redshift
        z = (sv * (1+z_guess))/c + z_guess
        z_sigma = (sv_sigma/c) * (1 + z_guess)

        # shift wavelengths to restframe
        restwave = wave / (1+z)
        restwave_sigma = wave * z_sigma / (1 + z)**2

        if bokeh:
            for y,x in zip(inds[0],inds[1]):
                wavecube[np.arange(len(restwave)),y,x] = restwave


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
            badbins['Infinite W'].append(ID)
            ewmap[w] = -999
            continue
        
        if not np.isfinite(W_sigma):
            badbins['Infinite EW_err'].append(ID)
            ewmap_unc[w] = -999
            continue
        
        ewmap[w] = W
        ewmap_unc[w] = W_sigma

        if not mask_EW:
            ewmap_mask[w] = 1


    logging.info('Creating plots...')
    flatew = ewmap.flatten()
    w = (flatew != 0) & (flatew != -999.0)
    flatewcleaned = flatew[w]
    
    bin_width = 3.5 * np.std(flatewcleaned) / (flatewcleaned.size ** (1/3))
    nbins = (max(flatewcleaned) - min(flatewcleaned)) / bin_width
    
    plt.hist(flatewcleaned,bins=int(nbins),color='k')
    plt.xlim(-5,5)
    plt.xlabel(r'$\mathrm{EW_{Na\ D}\ (\AA)}$')
    plt.ylabel(r'$N_{\mathrm{spax}}$')
    
    im2name = f"{args.galname}-EW_distribution.{args.imgftype}"
    output = os.path.join(savepath,im2name)
    plt.savefig(output,bbox_inches='tight',dpi=150)
    logging.info(f"EW distriubtion plot saved to {output}")
    plt.close()
    
    
    plotmap = np.copy(ewmap)
    plotmap[(plotmap==0) | (plotmap<vmin)] = np.nan
    
    nvmax = round(np.median(plotmap[np.isfinite(plotmap)]) + 3 * np.std(plotmap[np.isfinite(plotmap)]),1)

    im = plt.imshow(plotmap,origin='lower',cmap='rainbow',vmin=vmin,vmax=nvmax,
           extent=[32.4, -32.6,-32.4, 32.6])
    plt.xlabel(r'$\Delta \alpha$ (arcsec)')
    plt.ylabel(r'$\Delta \delta$ (arcsec)')
    
    ax = plt.gca()
    ax.set_facecolor('lightgray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im,cax=cax,label=r'$\mathrm{EW_{Na\ D}\ (\AA)}$')
    #plt.colorbar(label=r'$\mathrm{EW_{Na\ D}\ (\AA)}$',fraction=0.0465, pad=0.01)    
    
    im1name = f"{args.galname}-EW_map.{args.imgftype}"
    output = os.path.join(savepath,im1name)
    plt.savefig(output,bbox_inches='tight',dpi=200)
    logging.info(f"EW map plot saved to {output}")
    plt.close()
    


    plotmap = np.copy(ewmap)
    w = plotmap != -999.0
    med = np.median(plotmap[w])
    std = np.std(plotmap[w])

    nvmin = med - 4 * std
    nvmax = med + 4 * std

    im = plt.imshow(plotmap,origin='lower',cmap='rainbow',vmin=nvmin,vmax=nvmax,
           extent=[32.4, -32.6,-32.4, 32.6])
    plt.xlabel(r'$\Delta \alpha$ (arcsec)')
    plt.ylabel(r'$\Delta \delta$ (arcsec)')
    
    ax = plt.gca()
    ax.set_facecolor('lightgray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im,cax=cax,label=r'$\mathrm{EW_{Na\ D}\ (\AA)}$')
    #plt.colorbar(label=r'$\mathrm{EW_{Na\ D}\ (\AA)}$',fraction=0.0465, pad=0.01)    
    
    im1name = f"{args.galname}-EW_map_db.{args.imgftype}"
    output = os.path.join(savepath,im1name)
    plt.savefig(output,bbox_inches='tight',dpi=200)
    logging.info(f"EW map plot saved to {output}")
    plt.close()

    if args.bokeh:
        logging.info("Creating BOKEH plot.")
        keyword = f"{args.galname}-EW-bokeh"
        make_bokeh_map(flux, model, ivar, wavecube, ewmap, spatial_bins, savepath, keyword)

    return ewmap, ewmap_mask, ewmap_unc, badbins


def get_args():

    
    parser = argparse.ArgumentParser(description="A script to create an equivalent width map of ISM Na I for beta-corrected DAP outputs.")
    
    parser.add_argument('galname',type=str,help='Input galaxy name.')
    parser.add_argument('bin_method',type=str,help='Input DAP patial binning method.')
    parser.add_argument('--imgftype', type=str, help="Input filetype for output map plots. [pdf/png]", default = "pdf")
    parser.add_argument('--redshift',type=str,help='Input galaxy redshift guess.',default=None)
    parser.add_argument('--bokeh', type=bool, help="Input [True/False] for creating a Bokeh interactive plot.", default = False)
    
    return parser.parse_args()



def main(args):
    ## Full path of the script directory
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plt.style.use(os.path.join(script_dir,"figures.mplstyle"))
    ## Full path of the DAP outputs
    data_dir = os.path.join(script_dir, "data")

    ## Path to the specific galaxy and binning method input
    cube_dir = os.path.join(data_dir, "dap_outputs" ,f"{args.galname}-{args.bin_method}")
    cube_dir_bc = os.path.join(cube_dir,"BETA_CORR")
    check_filepath(cube_dir_bc, mkdir=False)
    
    ## Path to save the plots
    savepath = os.path.join(script_dir,"figures",'EW_map')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    ## All fits files within cube path
    fils = glob(os.path.join(cube_dir_bc,'**','*.fits'),recursive=True)
    
    ## Finds the logcube and map files from the DAP output
    cubefil = None
    mapfil = None
    for fil in fils:
        if 'LOGCUBE' in fil:
            cubefil = fil
        if 'MAPS' in fil:
            mapfil = fil
    
    if cubefil is None:
        raise ValueError(f'Cube File not found in {cube_dir_bc}')
    if mapfil is None:
        raise ValueError(f'maps File not found in {cube_dir_bc}')
    
    
    redshift = args.redshift

    
    if redshift is None:
        ## Get the redshift guess from the .ini file
        config_dir = os.path.join(data_dir,cube_dir,"config")
        ini_fil = glob(f"{config_dir}/*.ini")
        config_fil = ini_fil[0]
        if len(ini_fil)>1:
            warnings.warn(f"Multiple configuration files found in {config_dir}.",UserWarning)
            for fil in ini_fil:
                if 'cleaned' in fil:
                    config_fil = fil

            print(f"Defauling configuration file to {config_fil}.")


        config = configparser.ConfigParser()
        parsing = True
        while parsing:
            try:
                config.read(config_fil)
                parsing = False
            except configparser.Error as e:
                print(f"Error parsing file: {e}")
                print(f"Cleaning {config_fil}")
                clean_ini_file(config_fil, overwrite=True)


        redshift = config['default']['z']
        print(f"Redshift z={redshift} found in {config_fil}.")
    

    EW, EW_sigma, EW_mask, bbins = make_EW_map(cubefil,mapfil,redshift,savepath)
    
    mapspath = os.path.join(data_dir,cube_dir,"maps")
    if not os.path.exists(mapspath):
        os.mkdir(mapspath)

    savefits = os.path.join(mapspath,f"{cube_dir}_EW-maps.fits")

    logging.info(f"Writing data...")
    primary = fits.PrimaryHDU()

    hdu1 = fits.ImageHDU(EW, name="EW")
    hdu1.header['DESC'] = "ISM Na I equivalent width map"
    hdu1.header['UNITS'] = "Angstrom"
    
    hdu2 = fits.ImageHDU(EW_sigma, name="EW_SIGMA")
    hdu2.header['DESC'] = "Propagated error on the EW"
    hdu2.header['UNITS'] = "Angstrom"

    hdu3 = fits.ImageHDU(EW_mask, name="EW_MASK")
    hdu3.header['DESC'] = "Mask where EW values should be ignored"

    cols = []
    for key,values in bbins.items():
        cols.append(fits.Column(name=key, format='J', array = values))

    tablehdu = fits.BinTableHDU.from_columns(cols, name="BIN_IGNORES")
    tablehdu.header['DESC'] = "Spatial bin IDs of masked bins"

    hdul = fits.HDUList([primary,hdu1,hdu2,hdu3,tablehdu])
    hdul.writeto(savefits,overwrite=True)
    logging.info(f'Equivalent Width data written to {savefits}')

    logging.info(f"EW for {args.galname} finished with the following rejections:")
    keys = list(bbins.keys())
    print(f"Poor stellar continuum/emline fits: {len(bbins[keys[0]])} bins.")
    print(f"Poor stellar kinematic fits: {len(bbins[keys[1]])} bins.")
    print(f"Stellar velocity > 4 sigma: {len(bbins[keys[2]])} bins.")
    print(f"Non finite EW: {bbins[keys[3]]} bins.")
    
if __name__ == "__main__":
    args = get_args()
    if args.imgftype == "pdf" or args.imgftype == "png":
        pass
    else:
        raise ValueError(f"{args.imgftype} not a valid value for the output image filetype.\nAccepted formats [pdf/png]\nDefault: pdf")
    main(args)