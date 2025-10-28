import os
import numpy as np
from astropy.io import fits
import argparse
from modules import defaults, file_handler
from scipy.stats import iqr
from modules import util
from astropy.wcs import WCS
import astropy.units as u



def cube_from_DAP(galname, bin_method, wave_slice = None, verbose = False):
    util.sys_message('Preparing to create H_alpha CUBE from DAP outputs', color='yellow', verbose=verbose)
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    redshift = datapath_dict['Z']

    # open the data and get flux, continuum model, wavelength arr, and flux header
    with fits.open(datapath_dict['LOGCUBE']) as hdu:
        flux = hdu['flux'].data
        fluxheader = hdu['flux'].header
        stellar_continuum = hdu['model'].data
        wave = hdu['wave'].data

    # open maps and get the stellar velocity, dispersion, and masks for both
    with fits.open(datapath_dict['MAPS']) as hdul:
        stellar_vel = hdul['stellar_vel'].data
        stellar_vel_mask = hdul['stellar_vel_mask'].data

        stellar_sigma = hdul['stellar_sigma'].data
        stellar_sigma_mask = hdul['stellar_sigma_mask'].data

    # handle stellar velocity masks
    stellar_vel_mask = util.spec_mask_handler(stellar_vel_mask).astype(bool)
    stellar_sigma_mask = util.spec_mask_handler(stellar_sigma_mask).astype(bool)

    # calc mean vel dispersion
    v_disp = np.mean(stellar_sigma[~stellar_sigma_mask]) #(iqr(stellar_vel.flatten(), rng=(0.001,99.999)))/2

    # calc max and std of line-of-sight stellar velocity
    v_rot = np.max(abs(stellar_vel[~stellar_vel_mask]))
    v_rot_sigma = np.std(abs(stellar_vel[~stellar_vel_mask]))

    util.sys_message(f'Velocity Rotation (max) {v_rot}', color='yellow', verbose=verbose)
    util.sys_message(f'Velocity Dispersion (mean) {v_disp}', color='yellow', verbose=verbose)


    # if a wavelength slice is input, handle it
    if wave_slice is not None:
        if not isinstance(wave_slice, tuple) or len(wave_slice) != 2:
            raise ValueError(f"Input `wave_slice` must be a tuple of two wavelength values defining wavelength range boundaries")
        if wave_slice[1] <= wave_slice[0]:
            raise ValueError(f"values of `wave_slice` must be in ascending order")
        i1, i2 = np.argmin(abs(wave - wave_slice[0])), np.argmin(abs(wave - wave_slice[1]))

        wave = wave[i1:i2]
        flux_select = flux[i1:i2, :, :]
        stellar_continuum_select = stellar_continuum[i1:i2, :, :]

    # if no wave slice input, slice by the v_rot max around observed halpha:
    else:
        halpha = 6562.8 * (1 + redshift) # angstrom, observed
        c = 2.998e5
        rot_lambda = (v_rot / c) * halpha
        i1, i2 = np.argmin(abs(wave -  halpha - rot_lambda)), np.argmin(abs(wave - halpha + rot_lambda))

        wave = wave[i1:i2]
        flux_select = flux[i1:i2, :, :]
        stellar_continuum_select = stellar_continuum[i1:i2, :, :]

    # subtract the stellar continuum from the flux
    flux_subtract = flux_select - stellar_continuum_select
    
    util.sys_message(f'Slicing cube to {wave[0]} - {wave[-1]} angstrom; '
          f'rest-frame : {wave[0] / (1+redshift)} - {wave[-1] / (1+redshift)} angstrom', color='yellow', verbose=verbose)
    
    util.sys_message(f"New Cube DIMS: {flux_subtract.shape}", color='yellow', verbose=verbose)
    # copy the flux header and update the wavelength WCS values
    newheader = fluxheader.copy()

    newheader['NAXIS3'] = (len(wave), 'Number of wavelength pixels')
    newheader['CRVAL3'] = (wave[0], '[angstrom] Coordinate value at reference point')
    newheader['CRPIX3'] = (1, 'Pixel coordinate of reference point')
    newheader['PC3_3'] = np.diff(wave)[0], 'Coordinate transformation matrix element'
    newheader['CDELT3'] = (1.0, '[angstrom] Coordinate increment at reference point')
    newheader['CTYPE3'] = ('WAVE-LOG', 'Vacuum wavelength (logarithmic)')
    newheader['CUNIT3'] = 'angstrom'
    newheader['CRDER3'] = (fluxheader['CRDER3'] / 1e10, '[angstrom] random error in coordinate')

    # get galaxy info from config file and add galaxy info into header

    configuration = file_handler.parse_config(datapath_dict['CONFIG'], verbose=verbose)
    pa = configuration['pa']
    reff = configuration['reff']
    ebvgal = configuration['ebvgal']
    ell = configuration['ell']

    newheader['V_DISP'] = (v_disp, 'Mean velocity dispersion (km/s)')
    newheader['V_ROT'] = (v_rot, 'Maximum rotational velocity (km/s)')
    newheader['VROT_STD'] = (v_rot_sigma, 'Standard dev of rotational velocity (km/s)')
    newheader['REDSHIFT'] = (redshift, 'Systemic redshift')
    newheader['PA'] = (float(pa), 'Position angle (deg)')
    newheader['R_EFF'] = (float(reff), 'Effective radius (arcsec?)')
    newheader['ELL'] = (float(ell), 'Ellipticity 1 - b/a')
    newheader['EBVGAL'] = (float(ebvgal), 'E(B-V) Milky Way dust reddening (mag)')
    newheader['EXTNAME'] = ('PRIMARY', 'FITS HDU extension name')
    newheader.add_comment(f'Cube sliced by {wave[0]} - {wave[-1]} angstrom')
    newheader.add_comment(f'Cube sliced by {wave[0] / (1+redshift)} - {wave[-1] / (1+redshift)} angstrom (rest)')

    # write the fits data
    local_data_path = defaults.get_data_path('local')
    barolo_path = os.path.join(local_data_path, '3dbarolo')
    os.makedirs(barolo_path, exist_ok=True)
    outpath = os.path.join(barolo_path, f'{galname}-{bin_method}-Halpha_cube.fits')

    hdul = fits.PrimaryHDU(data=flux_subtract, header=newheader)
    hdul.writeto(outpath, overwrite=True)
    util.verbose_print(verbose, f"BBarolo H alpha cube written to {outpath}")

    

def cube_from_ESO(galname, bin_method, wave_slice = None, primary_only=True, verbose=False):
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
    #i1, i2 = np.argmin(abs(restframe - 6586)), np.argmin(abs(restframe - 6605))
    i1, i2 = np.argmin(abs(wavelengths - 6586)), np.argmin(abs(wavelengths - 6605))

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
    new_header.add_comment(f'Cube sliced by {wavelengths[i1]} - {wavelengths[i2]} angstrom')
    new_header.add_comment(f'Cube sliced by {restframe[i1]} - {restframe[i2]} angstrom rest-frame')
    print(f'Cube sliced by {wavelengths[i1]} - {wavelengths[i2]} angstrom')
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
    util.sys_warnings(f"barolo.py currently not functioning.... skipping")
    return
    if not os.path.exists(barolo_path) or not os.path.isfile(outpath):
        make_ha_cube(galname, bin_method, verbose=verbose)
    else:
        print('Barolo cube exists. Skipping...')


def get_args():
    parser = argparse.ArgumentParser(description="A script to handle the HII region collaboration.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)
    parser.add_argument('--eso', help = "Flag to specify creating the Halpha cube from the reduced ESO cube (default: False). If False," \
    " uses DAP cube instead", action='store_true', default=False)

    return parser.parse_args()

def main(args):
    galname = args.galname
    bin_method = args.bin_method
    verbose = args.verbose
    if args.eso:
        cube_from_ESO(galname, bin_method, verbose=verbose)
    else:
        cube_from_DAP(galname, bin_method, wave_slice=(6585, 6605), verbose=verbose)


if __name__ == "__main__":
    args = get_args()
    main(args)