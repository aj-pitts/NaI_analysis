import numpy as np
from astropy.io import fits
from tqdm import tqdm
from modules import util

def redshift_map_dict(galaxy_redshift, mapfile, verbose = False):
    maphdu = fits.open(mapfile)

    spatial_bins = maphdu['binid'].data[0]
    stellar_vel = maphdu['stellar_vel'].data
    stellar_vel_ivar = maphdu['stellar_vel_ivar'].data
    stellar_vel_mask = maphdu['stellar_vel_mask'].data

    c = 2.998e5

    items = np.unique(spatial_bins)[1:]
    iterator = tqdm(items, desc="Constructing Redshift Field") if verbose else items

    zmap = np.zeros_like(spatial_bins) - 1.0
    zmap_error = np.zeros_like(spatial_bins)
    zmap_mask = np.zeros_like(spatial_bins)

    for ID in iterator:
        w = ID == spatial_bins
        ny, nx = np.where(w)
        y, x = ny[0], nx[0]

        bin_check = util.check_bin_ID(ID, spatial_bins, DAPPIXMASK_list=[stellar_vel_mask], stellar_velocity_map=stellar_vel)
        zmap_mask[w] = bin_check

        sv = stellar_vel[y,x]
        sv_error = 1 / np.sqrt(stellar_vel_ivar[y,x])
        
        z = (sv * (1 + galaxy_redshift)) / c + galaxy_redshift
        z_error = (sv_error / c) * (1 + galaxy_redshift)

        if not np.isfinite(z) or not np.isfinite(z_error):
            if not np.isfinite(z):
                zmap_mask[w] = 4
            
            if not np.isfinite(z_error):
                zmap_mask[w] = 5
            continue

        zmap[w] = z
        zmap_error[w] = z_error

    return {'Redshift Map':zmap, 'Redshift Map Mask':zmap_mask, 'Reshift Map Error':zmap_error}