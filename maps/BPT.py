import numpy as np
from typing import Dict, Optional, Union, Tuple
import warnings

from modules import file_handler, util, defaults
from astropy.io import fits
from tqdm import tqdm
import os

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
        'liners': np.array([51, 54, 82]), # dark blue
        'liners': np.array([255, 38, 91]), # light red
        'unk': np.array([0, 0, 0])
    }
    color_hex = {
        'Star-Forming': '#77a9d4', # blue
        'Composite': '#ebebf0', # white
        'Seyfert': '#fad12c', # yellow
        'Liners': '#333652', # dark blue
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


def bpt_classify(h_alpha: float, h_beta: float, oiii_5007: float, nii_6584: float, 
                 oi_6300: Optional[float] = None, sii_6717: Optional[float] = None, 
                 sii_6731: Optional[float] = None, flux_errors: Optional[Dict[str, float]] = None,
                 snr_threshold: float = 3.0) -> Dict[str, Union[str, float, bool]]:
    """
    Classify galaxies using BPT (Baldwin-Phillips-Terlevich) emission line diagnostics.
    
    Uses the most modern and robust thresholds from:
    - Kauffmann et al. (2003) for star-formation/composite boundary
    - Kewley et al. (2001) for theoretical maximum starburst line
    - Schawinski et al. (2007) for Seyfert/LINER separation
    
    Parameters:
    -----------
    h_alpha : float
        H-alpha flux (6562.8 Å)
    h_beta : float  
        H-beta flux (4861.3 Å)
    oiii_5007 : float
        [OIII] 5007 Å flux
    nii_6584 : float
        [NII] 6584 Å flux
    oi_6300 : float, optional
        [OI] 6300 Å flux (for BPT-OI diagram)
    sii_6717 : float, optional
        [SII] 6717 Å flux (for BPT-SII diagram)  
    sii_6731 : float, optional
        [SII] 6731 Å flux (for BPT-SII diagram)
    flux_errors : dict, optional
        Dictionary of flux uncertainties with same keys as flux parameters
    snr_threshold : float, default=3.0
        Minimum signal-to-noise ratio for reliable classification
        
    Returns:
    --------
    dict : Classification results containing:
        - 'classification': Overall classification ('Star-forming', 'Composite', 'Seyfert', 'LINER', 'Ambiguous')
        - 'nii_bpt': NII-BPT diagram classification
        - 'sii_bpt': SII-BPT diagram classification (if SII data available)  
        - 'oi_bpt': OI-BPT diagram classification (if OI data available)
        - 'log_nii_ha': log([NII]/Hα) ratio
        - 'log_oiii_hb': log([OIII]/Hβ) ratio
        - 'reliable': Boolean indicating if SNR criteria are met
        - 'balmer_decrement': Hα/Hβ ratio for extinction assessment
    """
    
    # Initialize results dictionary
    results = {
        'classification': 'Unclassified',
        'nii_bpt': 'Unclassified', 
        'sii_bpt': 'Unclassified',
        'oi_bpt': 'Unclassified',
        'log_nii_ha': np.nan,
        'log_oiii_hb': np.nan,
        'reliable': True,
        'balmer_decrement': np.nan
    }
    
    # Check for non-positive fluxes
    required_fluxes = [h_alpha, h_beta, oiii_5007, nii_6584]
    if any(flux <= 0 for flux in required_fluxes):
        warnings.warn("Non-positive flux values detected. Classification may be unreliable.")
        results['reliable'] = False
        return results
    
    # Calculate Balmer decrement for extinction check
    balmer_decrement = h_alpha / h_beta
    results['balmer_decrement'] = balmer_decrement
    
    # Warn if Balmer decrement suggests significant extinction
    if balmer_decrement < 2.5:
        warnings.warn(f"Low Balmer decrement ({balmer_decrement:.2f}) suggests high extinction or measurement issues.")
    
    # Check signal-to-noise ratios if errors provided
    if flux_errors is not None:
        snr_checks = []
        flux_dict = {
            'h_alpha': h_alpha, 'h_beta': h_beta, 
            'oiii_5007': oiii_5007, 'nii_6584': nii_6584
        }
        
        for line, flux in flux_dict.items():
            if line in flux_errors and flux_errors[line] > 0:
                snr = flux / flux_errors[line]
                snr_checks.append(snr >= snr_threshold)
            else:
                snr_checks.append(True)  # Assume good if no error given
                
        if not all(snr_checks):
            results['reliable'] = False
            warnings.warn(f"Some lines below SNR threshold of {snr_threshold}")
    
    # Calculate primary line ratios for NII-BPT diagram
    log_nii_ha = np.log10(nii_6584 / h_alpha)
    log_oiii_hb = np.log10(oiii_5007 / h_beta)
    
    results['log_nii_ha'] = log_nii_ha
    results['log_oiii_hb'] = log_oiii_hb
    
    # NII-BPT Classification
    nii_class = classify_nii_bpt(log_nii_ha, log_oiii_hb)
    results['nii_bpt'] = nii_class
    
    # SII-BPT Classification (if data available)
    if sii_6717 is not None and sii_6731 is not None:
        if sii_6717 > 0 and sii_6731 > 0:
            sii_total = sii_6717 + sii_6731
            log_sii_ha = np.log10(sii_total / h_alpha)
            sii_class = classify_sii_bpt(log_sii_ha, log_oiii_hb)
            results['sii_bpt'] = sii_class
    
    # OI-BPT Classification (if data available)  
    if oi_6300 is not None:
        if oi_6300 > 0:
            log_oi_ha = np.log10(oi_6300 / h_alpha)
            oi_class = classify_oi_bpt(log_oi_ha, log_oiii_hb)
            results['oi_bpt'] = oi_class
    
    # Determine overall classification using hierarchical approach
    results['classification'] = determine_overall_classification(results)
    
    return results


def classify_nii_bpt(log_nii_ha: float, log_oiii_hb: float) -> str:
    """Classify using NII-BPT diagram with modern thresholds."""
    
    # Kewley et al. (2001) theoretical maximum starburst line
    # log([OIII]/Hβ) = 0.61 / (log([NII]/Hα) - 0.05) + 1.3
    def kewley_line(x):
        return 0.61 / (x - 0.05) + 1.3
    
    # Kauffmann et al. (2003) empirical star-formation line  
    # log([OIII]/Hβ) = 0.61 / (log([NII]/Hα) - 0.47) + 1.19
    def kauffmann_line(x):
        return 0.61 / (x - 0.47) + 1.19
    
    # Schawinski et al. (2007) Seyfert/LINER separation
    # log([OIII]/Hβ) = 1.05 * log([NII]/Hα) + 0.45
    seyfert_liner_slope = 1.05
    seyfert_liner_intercept = 0.45
    
    # Apply validity ranges for the demarcation lines
    if log_nii_ha < -3.0 or log_nii_ha > 1.0:
        return 'Out of range'
    
    try:
        kauffmann_threshold = kauffmann_line(log_nii_ha)
        kewley_threshold = kewley_line(log_nii_ha)
    except (ZeroDivisionError, OverflowError):
        return 'Ambiguous'
    
    # Classification logic
    if log_oiii_hb < kauffmann_threshold:
        return 'Star-forming'
    elif kauffmann_threshold <= log_oiii_hb < kewley_threshold:
        return 'Composite'  
    else:  # Above Kewley line
        seyfert_liner_threshold = seyfert_liner_slope * log_nii_ha + seyfert_liner_intercept
        if log_oiii_hb > seyfert_liner_threshold:
            return 'Seyfert'
        else:
            return 'LINER'


def classify_sii_bpt(log_sii_ha: float, log_oiii_hb: float) -> str:
    """Classify using SII-BPT diagram."""
    
    # Kewley et al. (2001) starburst line for SII
    # log([OIII]/Hβ) = 0.72 / (log([SII]/Hα) - 0.32) + 1.30
    def kewley_sii_line(x):
        return 0.72 / (x - 0.32) + 1.30
    
    # Seyfert/LINER separation (Kewley et al. 2006)
    # log([OIII]/Hβ) = 1.89 * log([SII]/Hα) + 0.76
    seyfert_liner_slope = 1.89
    seyfert_liner_intercept = 0.76
    
    if log_sii_ha < -2.0 or log_sii_ha > 1.0:
        return 'Out of range'
    
    try:
        kewley_sii_threshold = kewley_sii_line(log_sii_ha)
    except (ZeroDivisionError, OverflowError):
        return 'Ambiguous'
    
    if log_oiii_hb < kewley_sii_threshold:
        return 'Star-forming'
    else:
        seyfert_liner_threshold = seyfert_liner_slope * log_sii_ha + seyfert_liner_intercept
        if log_oiii_hb > seyfert_liner_threshold:
            return 'Seyfert'
        else:
            return 'LINER'


def classify_oi_bpt(log_oi_ha: float, log_oiii_hb: float) -> str:
    """Classify using OI-BPT diagram."""
    
    # Kewley et al. (2001) starburst line for OI
    # log([OIII]/Hβ) = 0.73 / (log([OI]/Hα) + 0.59) + 1.33  
    def kewley_oi_line(x):
        return 0.73 / (x + 0.59) + 1.33
    
    # Seyfert/LINER separation
    # log([OIII]/Hβ) = 1.18 * log([OI]/Hα) + 1.30
    seyfert_liner_slope = 1.18
    seyfert_liner_intercept = 1.30
    
    if log_oi_ha < -2.5 or log_oi_ha > 0.5:
        return 'Out of range'
    
    try:
        kewley_oi_threshold = kewley_oi_line(log_oi_ha)
    except (ZeroDivisionError, OverflowError):
        return 'Ambiguous'
    
    if log_oiii_hb < kewley_oi_threshold:
        return 'Star-forming'
    else:
        seyfert_liner_threshold = seyfert_liner_slope * log_oi_ha + seyfert_liner_intercept
        if log_oiii_hb > seyfert_liner_threshold:
            return 'Seyfert'
        else:
            return 'LINER'


def determine_overall_classification(results: Dict) -> str:
    """
    Determine overall classification using hierarchical approach.
    Prioritizes NII-BPT, then uses SII and OI BPT for confirmation/refinement.
    """
    
    primary = results['nii_bpt']
    secondary = results['sii_bpt'] 
    tertiary = results['oi_bpt']
    
    # If primary classification is clear and reliable, use it
    if primary in ['Star-forming', 'Composite']:
        return primary
    
    # For AGN classifications, use additional diagrams for refinement
    if primary == 'Seyfert':
        # Check if other diagrams agree
        other_classes = [secondary, tertiary]
        agn_votes = sum(1 for c in other_classes if c in ['Seyfert', 'LINER'])
        
        if agn_votes >= 1:  # At least one other diagram confirms AGN
            # Use majority vote for AGN subtype
            seyfert_votes = sum(1 for c in [primary, secondary, tertiary] if c == 'Seyfert')
            liner_votes = sum(1 for c in [primary, secondary, tertiary] if c == 'LINER')
            
            return 'Seyfert' if seyfert_votes >= liner_votes else 'LINER'
        else:
            return 'Ambiguous'
    
    elif primary == 'LINER':
        # Similar logic for LINER
        other_classes = [secondary, tertiary]
        agn_votes = sum(1 for c in other_classes if c in ['Seyfert', 'LINER'])
        
        if agn_votes >= 1:
            seyfert_votes = sum(1 for c in [primary, secondary, tertiary] if c == 'Seyfert')
            liner_votes = sum(1 for c in [primary, secondary, tertiary] if c == 'LINER')
            
            return 'LINER' if liner_votes >= seyfert_votes else 'Seyfert'
        else:
            return 'Ambiguous'
    
    else:
        return 'Ambiguous'

def classify_galaxy_bpt(galname: str, bin_method: str, include_errors = False, verbose = False):
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    maps = fits.open(datapath_dict['MAPS'])

    spatial_bins = maps['BINID'].data[0]
    emlines_map = maps['EMLINE_GFLUX'].data

    unique_bins = np.unique(spatial_bins)
    iterator = tqdm(unique_bins, desc="Computing BPT classifications") if verbose else unique_bins

    classification_int = {
        'Unclassified':0,
        'Star-forming':1,
        'Composite':2,
        'Seyfert':3,
        'LINER':4,
        'Ambiguous':5
    }

    BPTmap = np.zeros_like(spatial_bins)
    for ID in iterator:
        if ID == -1:
            continue
        w = ID == spatial_bins
        ys, xs = np.where(w)
        y, x = ys[0], xs[0]

        emlines = emlines_map[:,y,x]
        Halpha = emlines[23]
        Hbeta = emlines[14]
        Oiii_5007 = emlines[16]
        Sii_6718 = emlines[25]
        Sii_6732 = emlines[26]
        Oi_6302 = emlines[20]
        Nii_6585 = emlines[24]

        classification = bpt_classify(h_alpha=Halpha, h_beta=Hbeta, oiii_5007=Oiii_5007, nii_6584=Nii_6585,
                                  oi_6300=Oi_6302, sii_6717=Sii_6718, sii_6731=Sii_6732)
        bpt_value = classification_int[classification['classification']]
        BPTmap[w] = bpt_value

    hduname = 'BPT'
    BPTheader = {
        hduname:{
            "DESC":(f"Map of identifiers for {galname} BPT emission line classifications",""),
            "CLASS":("MAP","Data format"),
            "Int_0":("Unclassified","Description of integer 0"),
            "Int_1":("Star Forming","Description of integer 1"),
            "Int_2":("Composite","Description of integer 2"),
            "Int_3":("Seyfert","Description of integer 3"),
            "Int_4":("LINER","Description of integer 4"),
            "Int_5":("Ambiguous","Description of integer 5"),
            "EXTNAME":(hduname, "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    }

    BPT_dict = {hduname : BPTmap}

    BPT_mapdict = file_handler.standard_map_dict(galname, BPT_dict, custom_header_dict=BPTheader)
    file_handler.write_maps_file(galname, bin_method, [BPT_mapdict], verbose=verbose)


if __name__ == "__main__":
    print('BPT -m execution not functional')