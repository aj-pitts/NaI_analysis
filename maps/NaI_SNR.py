import numpy as np
from astropy.io import fits
from tqdm import tqdm


def NaD_snr_map(galname, cube_fil, maps_fil, zmap, verbose=False):
    cube = fits.open(cube_fil)
    maps = fits.open(maps_fil)

    spatial_bins = cube['BINID'].data[0]
    fluxcube = cube['FLUX'].data
    ivarcube = cube['IVAR'].data
    wave = cube['WAVE'].data

    stellarvel = maps['STELLAR_VEL'].data

    snr_map = np.zeros_like(stellarvel)
    windows = [(5865, 5875), (5915, 5925)]

    items = np.unique(spatial_bins)
    iterator = tqdm(np.unique(spatial_bins), desc="Constructing S/N Map") if verbose else items
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


        snr_map[w] = np.median(flux_arr / sigma_arr)

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

    return {"NaI_SNR":snr_map}, snr_header

