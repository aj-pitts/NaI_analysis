import numpy as np
from astropy.io import fits
from tqdm import tqdm
from modules import file_handler


def NaD_snr_map(galname, bin_method, zmap = None, verbose=False, write_data = True):
    datapath_dict = file_handler.init_datapaths(galname, bin_method)

    cube = fits.open(datapath_dict['LOGCUBE'])
    maps = fits.open(datapath_dict['MAPS'])

    spatial_bins = cube['BINID'].data[0]
    fluxcube = cube['FLUX'].data
    ivarcube = cube['IVAR'].data
    wave = cube['WAVE'].data

    stellarvel = maps['STELLAR_VEL'].data

    if zmap is None:
        local = fits.open(datapath_dict['LOCAL'])
        zmap = local['redshift'].data

    snr_map = np.zeros_like(stellarvel)
    windows = [(5865, 5875), (5915, 5925)]

    items = np.unique(spatial_bins)
    iterator = tqdm(np.unique(spatial_bins)[1:], desc="Constructing S/N Map") if verbose else items
    for ID in iterator:
        w = ID == spatial_bins
        y_inds, x_inds = np.where(w)
        
        z_bin = zmap[w][0]
        wave_bin = wave / (1+z_bin)

        wave_window = (wave_bin>=windows[0][0]) & (wave_bin<=windows[0][1]) | (wave_bin>=windows[1][0]) & (wave_bin<=windows[1][1])
        wave_inds = np.where(wave_window)[0]
        
        flux_arr = fluxcube[wave_inds, y_inds[0], x_inds[0]]
        ivar_arr = ivarcube[wave_inds, y_inds[0], x_inds[0]]
        sigma_arr = 1/np.sqrt(ivar_arr)
        
        real = np.isfinite(flux_arr) & np.isfinite(sigma_arr) & (sigma_arr > 0)
        snr_map[w] = np.median(flux_arr[real] / sigma_arr[real])
    
    snr_map[~np.isfinite(snr_map)] = 0

    hduname = "NaI_SNR"

    snr_header = {
        hduname:{
            "DESC":(f"{galname} median S/N of NaI",""),
            "HDUCLASS":("MAP", "Data format"),
            "UNITS":("", "Unit of pixel values"),
            "EXTNAME":(hduname, "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    }

    snr_dict = {"NaI_SNR":snr_map}

    if write_data:
        snr_mapdict = file_handler.standard_map_dict(galname, snr_dict, custom_header_dict=snr_header)
        file_handler.write_maps_file(galname, bin_method, [snr_mapdict], verbose=verbose)
    else:
        return snr_dict, snr_header

