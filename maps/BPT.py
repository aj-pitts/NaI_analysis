import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob
import os
import argparse
import sys
import astropy.units as u

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules/")))
from util import check_filepath, clean_ini_file

def classify_ELR(elr_dict):
    if (elr_dict['oiii'] < 0.61 / (elr_dict['nii'] - 0.05) + 1.3) and (elr_dict['oiii'] < 0.72/(elr_dict['sii'] - 0.32) + 1.3) and (elr_dict['oiii'] < 0.73 / (elr_dict['oi'] + 0.59) + 1.33):
        return 'star-forming'
    
    elif (elr_dict['oiii'] >  0.61 / (elr_dict['nii'] - 0.05) + 1.3) and (elr_dict['oiii'] < 0.61 / (elr_dict['nii'] - 0.47) + 1.19):
        return 'composite'
    
    elif (elr_dict['oiii'] > 0.61 / (elr_dict['nii'] - 0.47) + 1.19) and (elr_dict['oiii'] > 0.72/(elr_dict['sii'] - 0.32) + 1.3) and (elr_dict['oiii'] > 0.73 / (elr_dict['oi'] + 0.59) + 1.33):
        return 'seyfert'
    
    elif (elr_dict['oi'] > - 0.59) and (elr_dict['oiii'] > 1.89 * elr_dict['sii'] + 0.76) and (elr_dict['oiii'] > 1.18 * elr_dict['oi'] + 1.3):
        return 'seyfert'
    
    elif (elr_dict['oiii'] > 0.61 / (elr_dict['nii'] - 0.47) + 1.19) and (elr_dict['oiii'] > 0.72/(elr_dict['sii'] - 0.32) + 1.3) and (elr_dict['oiii'] < 1.89 * elr_dict['sii'] + 0.76):
        return 'liners'
    
    elif (elr_dict['oi'] > -0.59) and (elr_dict['oiii'] < 1.18 * elr_dict['oi'] + 1.3):
        return 'liners'
    
    else:
        return 'unk'
    


def BPT(map_fil, fig_output):
    """
    Creates spaxel-level BPT diagrams for individual galaxies using DAP emissionn line flux map and a new map
    containing color-coded classifications of each spaxel. Classifications are determined by Kewley et al. 2006.
    """

    logging.info("Obtaining Maps.")
    Map = fits.open(map_fil)
    emlines = Map['EMLINE_SFLUX'].data
    binid = Map['BINID'].data[0]


    ha = emlines[23]
    hb = emlines[14]

    #oiii = emlines[15] #oiii 4690
    oiii = emlines[16] #oiii 5007 

    oiii_hb = np.log10(oiii/hb)

    sii = emlines[25] + emlines[26] # sii 6718,32

    sii_ha = np.log10(sii/ha)

    oi = emlines[20] # oi 6302
    #oi = emlines[21] # oi 6365

    oi_ha = np.log10(oi/ha)

    #nii = emlines[22] # nii 6549
    nii = emlines[24] # nii 6585

    nii_ha = np.log10(nii/ha)

    fake_niiha = np.linspace(np.min(nii_ha[np.isfinite(nii_ha)]),np.max(nii_ha[np.isfinite(nii_ha)]),10000)
    fake_siiha = np.linspace(np.min(sii_ha[np.isfinite(sii_ha)]),np.max(sii_ha[np.isfinite(sii_ha)]),10000)
    fake_oiha = np.linspace(np.min(oi_ha[np.isfinite(oi_ha)]),np.max(oi_ha[np.isfinite(oi_ha)]),10000)



    logging.info("Creating BPT ELRs plot.")
    fig, ax = plt.subplots(1,3,figsize=(12,4),sharey=True)


    ax[0].plot(nii_ha,oiii_hb,'v',color='k',ms=1,alpha=0.25)

    demarcation = lambda log_lr: 0.61 / (log_lr - 0.05) + 1.3
    classification = demarcation(fake_niiha)
    w = fake_niiha < 0.05
    ax[0].plot(fake_niiha[w], classification[w], color='r')
    ax[0].set_ylabel(r"$\mathrm{log([O\ III]/H \beta)}$")
    ax[0].set_xlabel(r"$\mathrm{log([N\ II]/H \alpha)}$")



    ax[1].plot(sii_ha,oiii_hb, 'v', color='k',ms=1,alpha=0.25)
    
    demarcation = lambda log_lr: 0.72 / (log_lr - 0.32) + 1.3
    classification = demarcation(fake_siiha)
    w = fake_siiha < 0.32
    ax[1].plot(fake_siiha[w], classification[w], color='r')
    ax[1].set_xlabel(r"$\mathrm{log([S\ II]/H \alpha)}$")


    ax[2].plot(oi_ha,oiii_hb, 'v', color='k',ms=1,alpha=0.25)

    demarcation = lambda log_lr: 0.73 / (log_lr - 0.59) + 1.33
    classification = demarcation(fake_oiha)
    w = fake_oiha < 0.59
    ax[2].plot(fake_oiha[w], classification[w], color='r')
    ax[2].set_xlabel(r"$\mathrm{log([O\ I]/H \alpha)}$")

    ax[0].set_xlim(np.min(nii_ha[np.isfinite(nii_ha)]),np.max(nii_ha[np.isfinite(nii_ha)]))
    ax[0].set_ylim(np.min(oiii_hb[np.isfinite(oiii_hb)]),np.max(oiii_hb[np.isfinite(oiii_hb)]))
    ax[1].set_xlim(np.min(sii_ha[np.isfinite(sii_ha)]),np.max(sii_ha[np.isfinite(sii_ha)]))
    ax[1].set_ylim(np.min(oiii_hb[np.isfinite(oiii_hb)]),np.max(oiii_hb[np.isfinite(oiii_hb)]))
    ax[2].set_xlim(np.min(oi_ha[np.isfinite(oi_ha)]),np.max(oi_ha[np.isfinite(oi_ha)]))
    ax[2].set_ylim(np.min(oiii_hb[np.isfinite(oiii_hb)]),np.max(oiii_hb[np.isfinite(oiii_hb)]))

    fig.text(0.05, 0.9, "(a)", fontsize=15, transform = ax[0].transAxes)
    fig.text(0.05, 0.9, "(b)", fontsize=15, transform = ax[1].transAxes)
    fig.text(0.05, 0.9, "(c)", fontsize=15, transform = ax[2].transAxes)


    fig.subplots_adjust(hspace=0,wspace=0)
    
    imname = os.path.join(fig_output,f"{args.galname}_ELRs.{args.imgftype}")
    fig.savefig(imname,bbox_inches='tight',dpi=300)
    logging.info(f"ELRs Plot saved to {imname}")
    
    logging.info("Creating classification map.")
    class_map = np.zeros(binid.shape, dtype=int)

    color_map = {
        'star-forming': np.array([119, 168, 212]), # blue
        'composite': np.array([235, 235, 240]), # white
        'seyfert': np.array([250, 209, 44]), # yellow
        'liners': np.array([255, 38, 91]), # light red
        'unk': np.array([0, 0, 0])
    }

    color_hex = {
        'Star-Forming': '#77a9d4', # blue
        'Composite': '#ebebf0', # white
        'Seyfert': '#fad12c', # yellow
        'Liners': '#ff265b' # light red
    }

    classification_value = {
        'star-forming':0,
        'composite':1,
        'seyfert':2,
        'liners':3,
        'unk':-1
    }

    data3d = np.ndarray((binid.shape[0], binid.shape[0], 3), dtype=int)  

    for ID in np.unique(binid):
        w = binid == ID
        ys, xs = np.where(binid == ID)

        elrs = {
            'oiii':np.median(oiii_hb[w]),
            'nii':np.median(nii_ha[w]),
            'sii':np.median(sii_ha[w]),
            'oi':np.median(oi_ha[w])
        }

        classification_str = classify_ELR(elrs)
        class_map[w] = classification_value[classification_str]
        data3d[ys, xs] = color_map[classification_str]

    fig, ax = plt.subplots(1)
    ax.imshow(data3d,origin='lower',extent=[32.4, -32.6,-32.4, 32.6])

    ax.set_xlabel(r'$\Delta \alpha$ (arcsec)')
    ax.set_ylabel(r'$\Delta \delta$ (arcsec)')


    shift = 0
    for cat in color_hex.keys():
        x = 1.025
        y = 0.95
        ax.text(x, y-shift, cat, fontsize='large', transform=ax.transAxes, bbox={'facecolor':color_hex[cat],'pad':5})
        shift += 0.075

    imname = os.path.join(fig_output, f"{args.galname}_BPT-Classification.{args.imgftype}")
    fig.savefig(imname,bbox_inches='tight',dpi=300)
    logging.info(f"BPT classification Map plot saved to {imname}")

    return class_map


def get_args():
    parser = argparse.ArgumentParser(description="A script to create BPT plots and a spaxel classification map from the DAP emline results of a beta-corrected galaxy.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('--imgftype', type=str, help="Input filetype for output map plots. [pdf/png]", default = "pdf")

    return parser.parse_args()



def main(args):
    logging.info("Intitalizing directories and paths.")
    # intialize directories and paths
    repodir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plt.style.use(os.path.join(repodir,"figures.mplstyle"))
    datapath = os.path.join(repodir,"data",f"{args.galname}")

    fig_output_path = os.path.join(repodir,"figures","BPT")
    check_filepath(fig_output_path)

    map_output_path = os.path.join(datapath,"maps")
    check_filepath(map_output_path)

    cube_path = os.path.join(datapath,"cube",f"{args.galname}-{args.bin_method}","BETA-CORR")
    check_filepath(cube_path,mkdir=False)

    cube_fils = glob(os.path.join(cube_path,"**","*.fits"),recursive=True)
    map_fil = None
    for fil in cube_fils:
        if 'MAPS' in fil:
            map_fil = fil
            break

    if map_fil is None:
        print("Glob found:\n")
        print(cube_fils)
        raise ValueError(f"Maps file not found in {cube_path}")    
    
    logging.info("Done.")
    
    

    map_output_fil = os.path.join(map_output_path, f"{args.galname}_BPT-ID-Map.fits")

    logging.info(f"Writing BPT classification IDs to {map_output_fil}")
    class_map = BPT(map_fil, fig_output_path)
    hdu = fits.PrimaryHDU(class_map)
    hdu.header['DESC'] = f"BPT Classification ID Map for galaxy {args.galname}"
    hdu.header['SF_ID'] = 0
    hdu.header['COMP_ID'] = 1
    hdu.header['SEYF_ID'] = 2
    hdu.header['LINR_ID'] = 3
    hdu.header['OTHR_ID'] = -1
    hdul = fits.HDUList([hdu])
    hdul.writeto(map_output_fil,overwrite=True)
    logging.info("Done.")



if __name__ == "__main__":
    args = get_args()
    if args.imgftype == "pdf" or args.imgftype == "png":
        pass
    else:
        raise ValueError(f"{args.imgftype} not a valid value for the output image filetype.\nAccepted formats [pdf/png]\nDefault: pdf")
    main(args)