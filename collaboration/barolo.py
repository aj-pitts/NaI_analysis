import os
import numpy as np
from astropy.io import fits
import argparse
from modules import defaults, file_handler
from scipy.stats import iqr
from modules import util

def get_args():
    parser = argparse.ArgumentParser(description="A script to handle the HII region collaboration.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)

    return parser.parse_args()

def make_ha_cube(galname, bin_method, primary_only=True, verbose=False):
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    redshift = datapath_dict['Z']
    mapfile = datapath_dict['MAPS']

    maps = fits.open(mapfile)
    stellar_vel = maps['stellar_vel'].data.flatten()
    v_disp = (iqr(stellar_vel.flatten(), rng=(16,84)))/2


    pipeline_path = defaults.get_data_path('pipeline')
    muse_cubes = os.path.join(pipeline_path, 'muse_cubes')
    galaxy_path = os.path.join(muse_cubes, galname)
    if not os.path.exists(galaxy_path):
        raise ValueError(f"Path does not exist: {galaxy_path}")
    
    filepath = os.path.join(galaxy_path, f"{galname}.fits")
    if not os.path.isfile(filepath):
        raise ValueError(f"Filepath does not exist: {filepath}")
    
    util.verbose_print(verbose, f"Creating H alpha cube from {filepath}")

    data = fits.open(filepath)

    header = data['data'].header

    crval = header['crval3']
    crpix = header['crpix3']
    deltaval = header['cd3_3']
    naxis = header['naxis3']
    wavelengths = crval + (np.arange(naxis) - (crpix - 1)) * deltaval
    
    restframe = wavelengths / (1 + redshift)
    halpha = 6562.8 # angstrom
    c = 2.998e5
    disp_lambda = (v_disp / c) * halpha
    
    i1, i2 = np.argmin(abs(restframe - (halpha - disp_lambda))), np.argmin(abs(restframe - (halpha + disp_lambda)))
    
    flux_cube = data['data'].data
    sliced_flux = flux_cube[i1:i2,:,:]

    new_header = header.copy()
    new_header['NAXIS3'] = i2 - i1
    new_header['CRVAL3'] = wavelengths[i1]
    new_header['CRPIX3'] = 1
    new_header['OBJECT'] = galname
    new_header['EXTNAME'] = 'PRIMARY'
    new_header.add_comment(f'H-alpha cube sliced around {halpha:.1f} angstrom rest-frame')
    new_header.add_comment(f'Velocity dispersion: {v_disp:.1f} km/s')
    new_header.add_comment(f'Redshift: {redshift:.6f}')

    if not primary_only:
        raise ValueError("flux and ivar HDUList not currently supported in barolo.py")
        ivar_cube = data['stat'].data
        sliced_ivar = ivar_cube[i1:i2,:,:]
        ivarhdu = fits.ImageHDU(data = sliced_ivar, name='IVAR')
        fluxhdu = fits.ImageHDU(data = sliced_flux, name='FLUX')

        hdul = fits.HDUList([fits.PrimaryHDU(), fluxhdu, ivarhdu])
    else:
        hdul = fits.PrimaryHDU(data=sliced_flux, header=new_header)

    
    local_data_path = defaults.get_data_path('local')
    barolo_path = os.path.join(local_data_path, '3dbarolo')
    os.makedirs(barolo_path, exist_ok=True)

    outpath = os.path.join(barolo_path, f'{galname}-{bin_method}-Halpha_cube.fits')
    hdul.writeto(outpath, overwrite=True)
    util.verbose_print(verbose, f"H alpha cube written to {outpath}")


def main(args):
    galname = args.galname
    bin_method = args.bin_method
    verbose = args.verbose
    
    datapath_dict = file_handler.init_datapaths(galname, bin_method, verbose=verbose)
    redshift = datapath_dict['Z']
    cubefile = datapath_dict['LOGCUBE']

    cube = fits.open(cubefile)
    stellar_vel = cube['stellar_vel'].data.flatten()
    v_disp = (iqr(stellar_vel.flatten(), rng=(16,84)))/2


    pipeline_path = defaults.get_data_path('pipeline')
    muse_cubes = os.path.join(pipeline_path, 'muse_cubes')
    galaxy_path = os.path.join(muse_cubes, galname)
    if not os.path.exists(galaxy_path):
        raise ValueError(f"Path does not exist: {galaxy_path}")
    
    filepath = os.path.join(galaxy_path, f"{galname}.fits")
    if not os.path.isfile(filepath):
        raise ValueError(f"Filepath does not exist: {filepath}")
    
    data = fits.open(filepath)

    header = data['data'].header

    crval = header['crval3']
    crpix = header['crpix3']
    deltaval = header['cd3_3']
    naxis = header['naxis3']
    wavelengths = crval + (np.arange(naxis) - (crpix - 1)) * deltaval
    
    restframe = wavelengths / (1 + redshift)
    halpha = 6562.8 # angstrom
    c = 2.998e5
    disp_lambda = (v_disp / c) * halpha
    
    i1, i2 = np.argmin(abs(restframe - (halpha - disp_lambda))), np.argmin(abs(restframe - (halpha + disp_lambda)))
    
    flux_cube = data['data'].data
    ivar_cube = data['stat'].data

    sliced_flux = flux_cube[i1:i2,:,:]
    sliced_ivar = ivar_cube[i1:i2,:,:]
    
    fluxhdu = fits.ImageHDU(data = sliced_flux, name='FLUX')
    ivarhdu = fits.ImageHDU(data= sliced_ivar, name='IVAR')

    hdul = fits.HDUList([fits.PrimaryHDU(), fluxhdu, ivarhdu])
    
    local_data_path = defaults.get_data_path('local')
    barolo_path = os.path.join(local_data_path, '3dbarolo')
    os.makedirs(barolo_path, exist_ok=True)

    outpath = os.path.join(barolo_path, f'{galname}-{bin_method}-Halpha_cube.fits')
    hdul.writeto(outpath, overwrite=True)


if __name__ == "__main__":
    args = get_args()
    main(args)