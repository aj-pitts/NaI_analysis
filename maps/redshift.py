import numpy as np
from astropy.io import fits
from tqdm import tqdm
from modules import util, file_handler

def redshift_map_dict(galname, bin_method, verbose = False, write_data = True):
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    maphdu = fits.open(datapath_dict['MAPS'])
    galaxy_redshift = datapath_dict['Z']

    spatial_bins = maphdu['binid'].data[0]
    stellar_vel = maphdu['stellar_vel'].data
    stellar_vel_ivar = maphdu['stellar_vel_ivar'].data
    stellar_vel_mask = maphdu['stellar_vel_mask'].data

    c = 2.998e5

    items = np.unique(spatial_bins)
    iterator = tqdm(items, desc="Constructing Redshift Field") if verbose else items

    zmap = np.zeros_like(spatial_bins) - 1.0
    zmap_error = np.zeros_like(spatial_bins)
    zmap_mask = np.zeros_like(spatial_bins)

    for ID in iterator:
        w = ID == spatial_bins
        if ID == -1:
            zmap_mask[w] = 1
            continue
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

    map_dict = {'Redshift Map':zmap, 'Redshift Map Mask':zmap_mask, 'Reshift Map Error':zmap_error}

    if write_data:
        redshift_mapdict = file_handler.standard_map_dict(galname, map_dict, HDU_keyword="REDSHIFT", IMAGE_units="")
        file_handler.write_maps_file(galname, bin_method, [redshift_mapdict], verbose=verbose, preserve_standard_order=True)

    else: 
        return map_dict