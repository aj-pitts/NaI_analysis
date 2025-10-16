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
    config = datapath_dict['CONFIG']

    with fits.open(mapfile) as maps:
        stellar_vel = maps['stellar_vel'].data
        stellar_vel_mask = maps['stellar_vel_mask'].data

        stellar_sigma = maps['stellar_sigma'].data
        stellar_sigma_mask = maps['stellar_sigma_mask'].data

    stellar_vel_mask = util.spec_mask_handler(stellar_vel_mask).astype(bool)
    stellar_sigma_mask = util.spec_mask_handler(stellar_sigma_mask).astype(bool)

    v_disp = np.mean(stellar_sigma[~stellar_sigma_mask]) #(iqr(stellar_vel.flatten(), rng=(0.001,99.999)))/2

    v_rot = np.max(abs(stellar_vel[~stellar_vel_mask]))
    v_rot_sigma = np.std(abs(stellar_vel[~stellar_vel_mask]))

    if verbose: 
        print(f'Velocity Rotation (max) {v_rot}')
        print(f'Velocity Dispersion (mean) {v_disp}')


    pipeline_path = defaults.get_data_path('pipeline')
    muse_cubes = os.path.join(pipeline_path, 'muse_cubes')
    galaxy_path = os.path.join(muse_cubes, galname)
    if not os.path.exists(galaxy_path):
        raise ValueError(f"Path does not exist: {galaxy_path}")
    
    muse_cube_path = os.path.join(galaxy_path, f"{galname}.fits")
    if not os.path.isfile(muse_cube_path):
        raise ValueError(f"MUSE_cube_path does not exist: {muse_cube_path}")
    
    util.verbose_print(verbose, f"Creating H alpha cube from {muse_cube_path}")

    with fits.open(muse_cube_path) as data:
        header = data['data'].header
        flux_cube = data['data'].data

    crval = header['crval3']
    crpix = header['crpix3']
    deltaval = header['cd3_3']
    naxis = header['naxis3']
    wavelengths = crval + (np.arange(naxis) - (crpix - 1)) * deltaval
    
    restframe = wavelengths / (1 + redshift)
    # halpha = 6562.8 # angstrom
    # c = 2.998e5
    # rot_lambda = (v_rot / c) * halpha
    
    # i1, i2 = np.argmin(abs(restframe - (halpha - rot_lambda))), np.argmin(abs(restframe - (halpha + rot_lambda)))
    i1, i2 = np.argmin(abs(restframe - 6586)), np.argmin(abs(restframe - 6605))

    sliced_flux = flux_cube[i1:i2,:,:]


    configuration = file_handler.parse_config(config, verbose=verbose)
    pa = configuration['pa']
    reff = configuration['reff']
    ebvgal = configuration['ebvgal']
    ell = configuration['ell']


    new_header = header.copy()
    new_header['NAXIS3'] = (i2 - i1, 'Number of wavelength pixels')
    new_header['CRVAL3'] = (wavelengths[i1], 'Wavelength at reference pixel (Angstrom)')
    new_header['CRPIX3'] = (1, 'Reference pixel along wavelength axis')
    new_header['V_DISP'] = (v_disp, 'Mean velocity dispersion (km/s)')
    new_header['V_ROT'] = (v_rot, 'Maximum rotational velocity (km/s)')
    new_header['VROT_STD'] = (v_rot_sigma, 'Standard deviation of rotational velocity (km/s)')
    new_header['REDSHIFT'] = (redshift, 'Systemic redshift')
    new_header['PA'] = (float(pa), 'Position angle (deg)')
    new_header['R_EFF'] = (float(reff), 'Effective radius')
    new_header['ELL'] = (float(ell), 'Ellipticity 1 - b/a')
    new_header['EBVGAL'] = (float(ebvgal), 'E(B-V) Milky Way dust reddening (mag)')
    new_header['EXTNAME'] = ('PRIMARY', 'FITS HDU extension name')
    new_header.add_comment(f'Cube sliced by {restframe[i1]} - {restframe[i2]} angstrom rest-frame')
    print(f'Cube sliced by {restframe[i1]} - {restframe[i2]} angstrom rest-frame')

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

def analysis_run(galname, bin_method, verbose = False):
    local_data_path = defaults.get_data_path('local')
    barolo_path = os.path.join(local_data_path, '3dbarolo')
    outpath = os.path.join(barolo_path, f'{galname}-{bin_method}-Halpha_cube.fits')
    if not os.path.exists(barolo_path) or not os.path.isfile(outpath):
        make_ha_cube(galname, bin_method, verbose=verbose)
    else:
        print('Barolo cube exists. Skipping...')

def main(args):
    galname = args.galname
    bin_method = args.bin_method
    verbose = args.verbose
    make_ha_cube(galname, bin_method, verbose=verbose)


if __name__ == "__main__":
    args = get_args()
    main(args)