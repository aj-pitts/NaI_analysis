import os
import numpy as np
from astropy.io import fits, ascii

import argparse
import sys
import pdb
import traceback

from scipy.optimize import curve_fit
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from modules import util, defaults, file_handler, plotter, inspect
from modules.util import verbose_print

import plotly.graph_objects as go
import plotly.io as pio

import cmasher as cmr

import string
import warnings


def plot_local_maps(galname, bin_method, verbose=False):
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    local_maps_path = datapath_dict['LOCAL']

    hdul = fits.open(local_maps_path)

    results_dir = defaults.get_fig_paths(galname, bin_method, subdir='results')
    maps_directory = os.path.join(results_dir,"maps")
    util.check_filepath(maps_directory, verbose=verbose)
    hist_directory = os.path.join(results_dir, "hists")
    util.check_filepath(hist_directory, verbose=verbose)

    util.verbose_print(verbose, "Creating individual map plots...")

    def figname(keyword):
        return f"{galname}-{bin_method}-{keyword}.pdf"
    def histname(keyword):
        return f"{galname}-{bin_method}-{keyword}_HIST.pdf"
    # redshift
    plotter.MAP_plotter(hdul['redshift'].data, r'$z$', maps_directory, figname('REDSHIFT'), mask_data=hdul['redshift_mask'].data, verbose=verbose, 
                        **dict(cmap=util.seaborn_palette('seismic')))
    plotter.HIST_plotter(hdul['redshift'].data, hdul['spatial_bins'].data, r'$z$', hist_directory, histname('REDSHIFT'), mask_data=hdul['redshift_mask'].data, verbose=verbose)

    # Na D S/N
    plotter.MAP_plotter(hdul['nai_snr'].data, r'$S/N_{\mathrm{Na\ D}}$', maps_directory, figname('NAI_SNR'), verbose=verbose, 
                        **dict(vmin=0, vmax=100, cmap='managua'))
    plotter.HIST_plotter(hdul['nai_snr'].data, hdul['spatial_bins'].data, r'$S/N_{\mathrm{Na\ D}}$', hist_directory, histname('NAI_SNR'), verbose=verbose)

    # Equiv Width
    plotter.MAP_plotter(hdul['ew_nai'].data, r'$\mathrm{EW_{Na\ D}}\ \left( \mathrm{\AA} \right)$', maps_directory, figname('EW_NAI'), 
                        mask_data = hdul['ew_nai_mask'].data, verbose=verbose, **dict(vmin=-0.5, vmax=2, cmap='rainbow'))
    plotter.HIST_plotter(hdul['ew_nai'].data, hdul['spatial_bins'].data, r'$\mathrm{EW_{Na\ D}}\ \left( \mathrm{\AA} \right)$', hist_directory, histname('EW_NAI'), 
                        mask_data = hdul['ew_nai_mask'].data, verbose=verbose)
    
    # Equiv Width Absorption Only
    plotter.MAP_plotter(hdul['ew_noem'].data, r'$\mathrm{EW_{Na\ D}}\ \left( \mathrm{\AA} \right)$', maps_directory, figname('EW_NOEM'), 
                        mask_data = hdul['ew_noem_mask'].data, verbose=verbose, **dict(vmin=-0.5, vmax=2, cmap='rainbow'))
    plotter.HIST_plotter(hdul['ew_noem'].data, hdul['spatial_bins'].data, r'$\mathrm{EW_{Na\ D}}\ \left( \mathrm{\AA} \right)$', hist_directory, histname('EW_NOEM'), 
                        mask_data = hdul['ew_noem_mask'].data, verbose=verbose)
    
    # Star Formation Rate Surface Density
    plotter.MAP_plotter(hdul['sfrsd'].data, r"$\mathrm{log \Sigma_{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spx^{-1}} \right)$", maps_directory,
                        figname('SFRSD'), mask_data = hdul['sfrsd_mask'].data, verbose=verbose, **dict(cmap='rainbow', vmin=-2.5, vmax=0,))
    plotter.HIST_plotter(hdul['sfrsd'].data, hdul['spatial_bins'].data, r"$\mathrm{log \Sigma_{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spx^{-1}} \right)$", hist_directory,
                        histname('SFRSD'), mask_data = hdul['sfrsd_mask'].data, verbose=verbose)
    
    # Na D centroid velocity
    plotter.MAP_plotter(hdul['v_nai'].data, r"$v_{\mathrm{cen}}\ \left( \mathrm{km\ s^{-1}} \right)$", maps_directory, figname('V_NaI'),
                        mask_data=hdul['v_nai_mask'].data, verbose=verbose, **dict(vmin=-250, vmax=250, cmap='seismic'))
    plotter.MAP_plotter(hdul['v_nai'].data, r"$v_{\mathrm{cen}}\ \left( \mathrm{km\ s^{-1}} \right)$", maps_directory, figname('V_NaI_unmasked'),
                        mask_data=(hdul['v_nai'].data == -999),verbose=verbose, **dict(vmin=-250, vmax=250, cmap='seismic'))
    plotter.HIST_plotter(hdul['v_nai'].data, hdul['spatial_bins'].data, r"$v_{\mathrm{cen}}\ \left( \mathrm{km\ s^{-1}} \right)$", hist_directory, histname('V_NaI'),
                        mask_data=hdul['v_nai_mask'].data, verbose=verbose)
    
    # extinction
    plotter.MAP_plotter(hdul['e(b-v)'].data, r"$E(B-V)$", maps_directory, figname('EXTINCTION'), verbose=verbose, **dict(vmin=0, vmax=1, cmap='rainbow'))

    plotter.plot_BPT_MAP(galname, bin_method, verbose=verbose)

    # metallicity
    # plotter.MAP_plotter(hdul['metallicity'].data, r"$12 + \mathrm{log\ O/H}$", maps_directory, figname('METALLICITY'),
    #                     mask_data=hdul['metallicity_mask'].data, verbose=verbose, **dict(cmap='rainbow'))
    hdul.close()

def velocity_vs_sfr(galname, bin_method, output_dir = None, pearson = True, contours = True,
                    radius_cbar = False, hists = False, verbose = False):
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    local_maps_path = datapath_dict['LOCAL']

    with fits.open(local_maps_path) as hdul:
        spatial_bins = hdul['SPATIAL_BINS'].data
        radius_map = hdul['RADIUS'].data

        vmap = hdul['V_NaI'].data
        vmap_mask = hdul['V_NaI_MASK'].data
        vmap_error = np.mean(hdul['V_NaI_ERROR'].data, axis=0)
        vfrac = hdul['v_nai_frac'].data

        sfrmap = hdul['SFRSD'].data
        sfrmap_mask = hdul['SFRSD_MASk'].data
        sfrmap_error = hdul['SFRSD_ERROR'].data

    ### TODO HANDLE ASYMMETRIC ERRORS
    ## mask out values
    mapmask = np.logical_or(vmap_mask.astype(bool), sfrmap_mask.astype(bool))
    w = (vmap == -999) | (sfrmap == -999)
    mask = np.logical_or(mapmask, w)

    _, bin_inds = np.unique(spatial_bins[~mask], return_index=True)

    sfrs = sfrmap[~mask][bin_inds]
    sfr_errors = sfrmap_error[~mask][bin_inds]

    velocities = vmap[~mask][bin_inds]
    velocity_errors = vmap_error[~mask][bin_inds]

    radii = radius_map[~mask][bin_inds]


    inflow = velocities > 0
    outflow = velocities <= 0

    sfrs_i = sfrs[inflow]
    velocities_i = velocities[inflow]
    errors_i = velocity_errors[inflow]

    sfrs_o = sfrs[outflow]
    velocities_o = velocities[outflow]
    errors_o = velocity_errors[outflow]

    ## colormap and errorbar style
    if radius_cbar:
        normalized = mcolors.Normalize(vmin = np.min(radii), vmax = np.max(radii))
        cmap = cm.Oranges
        colors = cmap(normalized(radii))
    else:
        normalized_o = mcolors.Normalize(vmin = -250, vmax = 0)
        cmap_o = cm.Blues_r
        colors_o = cmap_o(normalized_o(velocities_o))

        normalized_i = mcolors.Normalize(vmin = 0, vmax = 250)
        cmap_i = cm.Reds
        colors_i = cmap_i(normalized_i(velocities_i))



    scatterstyle = dict(
        marker = 'o',
        s = 1.9**2,
        linewidths = 0.4,
        edgecolors = 'k',
    )

    errorstyle = dict(
        linestyle = 'none',
        marker = 'o',
        ms = 1.9,
        lw = 1,
        markeredgecolor = 'k',
        markeredgewidth = 0.4,
        ecolor = 'k',
        elinewidth = 0.4,
        capsize = 1.5
    )

    ## scatter for masked "bad" values, errorbar for good values
    fig, ax = plt.subplots(figsize=(7, 7))
    # for sf, vel, c in zip(sfrs, velocities, colors):
    #     scatter = ax.scatter(sf, vel, facecolors=c, **scatterstyle)

    scatter_i = ax.scatter(sfrs_i, velocities_i, facecolors = colors_i, **scatterstyle)
    scatter_o = ax.scatter(sfrs_o, velocities_o, facecolors = colors_o, **scatterstyle)
    
    x_pos = 0.1
    y_pos = 0.9
    x_pos_data, y_pos_data = ax.transData.inverted().transform(ax.transAxes.transform((x_pos, y_pos)))
    ax.errorbar(x_pos_data, y_pos_data, yerr=np.mean(velocity_errors), fmt='none', color='black', capsize=3, elinewidth=1.5)


    #sm = cm.ScalarMappable(cmap=cmap, norm=normalized)
    #sm.set_array([])  # Dummy array for colorbar
    #cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    #cbar.set_label(label=r"$R / R_e$", rotation=270, labelpad=21)

    ax.set_xlim(min(np.min(sfrs_i), np.min(sfrs_o)),max(np.max(sfrs_i), np.max(sfrs_o)))
    ax.set_xlabel(r'$\mathrm{log\ \Sigma_{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spx^{-1}} \right)$')
    ax.set_ylabel(r'$v_{\mathrm{Na\ D}}\ \left( \mathrm{km\ s^{-1}} \right)$')
    ax.set_ylim(-350, 350)

    ## pearson rank test
    if pearson:
        pearson_result = pearsonr(sfrs, velocities)
        val = round(pearson_result[0], 1)
        ax.text(0.75, 0.85, fr'$\rho = {0.0 if val == -0.0 else val:.1f}$', fontsize=11, transform = ax.transAxes)

    if hists:
        # Create new axes for the histograms
        divider = make_axes_locatable(ax)
        ax_hist_top = divider.append_axes("top", size="20%", pad=0.1, sharex=ax)
        ax_hist_right = divider.append_axes("right", size="20%", pad=0.1, sharey=ax)

        # Plot histograms
        ax_hist_top.hist(sfrs, bins=75, alpha=.5, color='dimgray',)
        ax_hist_top.hist(sfrs, bins=75, alpha=1, color='k', histtype='step', linewidth=1.5)

        ax_hist_right.hist(velocities, bins=75, alpha=0.5, color='dimgray', orientation='horizontal')
        ax_hist_right.hist(velocities, bins=75, alpha=1, color='k', histtype='step', orientation='horizontal', linewidth=1.5)

        # Adjust labels and limits
        ax_hist_top.tick_params(axis="x", labelbottom=False)
        ax_hist_right.tick_params(axis="y", labelleft=False)
        ax_hist_right.set_ylim(-350,350)
        #ax_hist_top.set_ylabel("Count")
        #ax_hist_right.set_xlabel("Count")
        plt.tight_layout()

    if output_dir is None:
        results_dir = defaults.get_fig_paths(galname, bin_method, subdir='results')
        output_dir = os.path.join(results_dir, 'scatter')
        util.check_filepath(output_dir, verbose=verbose)

    outfil = os.path.join(output_dir, f'{galname}-{bin_method}-v_vs_sfr.pdf')
    plt.savefig(outfil, bbox_inches='tight')
    verbose_print(verbose, f"Velocity vs SFR fig saved to {outfil}")


    if contours:
        z = np.histogram2d(sfrs, velocities, bins=53, range=[[-2.5, 0], [-250, 250]])[0]

        fig = go.Figure(go.Contour(
            z=z.T,  # Values for the contour plot
            x=np.linspace(-2.5, 0, z.shape[1]),  # x data (horizontal axis)
            y=np.linspace(-250, 250, z.shape[0]),  # y data (vertical axis)
            colorscale='Greys',  # Color scale
            showscale=False
        ))

        # Set plot labels and title
        fig.update_layout(
            xaxis_title=r"$ \mathrm{log\ \Sigma_{SFR}} \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spx^{-1}} \right)$",
            yaxis_title=r"$ v_{\mathrm{Na\ D}}\ \left( \mathrm{km\ s^{-1}} \right) $",
            width=600,
            height=600,
            margin=dict(t=15, b=15, l=15, r=15)
        )

        fig.update_xaxes(
            range=[np.min(sfrs), np.max(sfrs)],
            linecolor='black',
            linewidth=2,
            showline=True,
            showgrid=True,
            ticks="inside",
            ticklen=8,
            tickmode="array",
            minor_ticks='inside', 
            minor_ticklen=3,
            mirror='all',
        )

        fig.update_yaxes(
            range=[-350, 350],
            linecolor='black',
            linewidth=2,
            showline=True,
            showgrid=True,
            ticks="inside",
            ticklen=8,
            tickmode="array",
            minor_ticks='inside',
            minor_ticklen=3,
            mirror='all',
        )
            
        outfil = os.path.join(output_dir, f'{galname}-{bin_method}-v_vs_sfr-contours.pdf')
        pio.write_image(fig, outfil) 
        verbose_print(verbose, f"Velocity vs SFR with contours saved to {outfil}")


def terminal_velocity(galname, bin_method, show = False, save = True, output_dir = None, radius_cbar = False, power_law = True, verbose = True):
    # open local maps fits file
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    local_maps_path = datapath_dict['LOCAL']

    with fits.open(local_maps_path) as hdul:
        # init relevant data
        spatial_bins = hdul['SPATIAL_BINS'].data
        radius_map = hdul['RADIUS'].data

        sfrmap = hdul['SFRSD'].data
        sfrmap_mask = hdul['SFRSD_MASK'].data
        sfrmap_error = hdul['SFRSD_ERROR'].data

        vterm = hdul['V_MAX_OUT'].data
        vterm_mask = hdul['V_MAX_OUT_MASK'].data
        vterm_error = np.mean(hdul['V_MAX_OUT_ERROR'].data, axis=0)


    combined_mask = np.logical_or(vterm_mask.astype(bool), sfrmap_mask.astype(bool))
    mask = combined_mask | (sfrmap == -999) | (vterm >= 400)

    masked_vterm = vterm[~mask]
    masked_vterm_error = vterm_error[~mask]

    masked_sfrmap = sfrmap[~mask]
    masked_sfrmap_error = sfrmap_error[~mask]

    masked_radius = radius_map[~mask]

    masked_bins = spatial_bins[~mask]
    _ , bin_inds = np.unique(masked_bins, return_index=True)

    terminal_velocities = masked_vterm[bin_inds]
    terminal_velocity_errors = masked_vterm_error[bin_inds]

    sfrs = masked_sfrmap[bin_inds]
    sfr_errors = masked_sfrmap_error[bin_inds]

    radii = masked_radius[bin_inds]

    if radius_cbar:
        normalized = mcolors.Normalize(vmin = np.min(radii), vmax = np.max(radii))
        cmap = cm.Oranges
        colors = cmap(normalized(radii))
    else:
        normalized = mcolors.Normalize(vmin = np.min(terminal_velocities), vmax = np.max(terminal_velocities))
        cmap = cm.Blues
        colors = cmap(normalized(terminal_velocities))
    
    style = dict(
        linestyle = 'none',
        marker = 'o',
        ms = 1.5,
        lw = 1.5,
        markeredgecolor = 'k',
        markeredgewidth = 0.6,
        ecolor = 'k',
        elinewidth = .6,
        capsize = 2
    )  

    fig, ax = plt.subplots(figsize=(7,7))

    xmin, xmax = sfrs.min(), sfrs.max()
    items = zip(terminal_velocities, sfrs, colors)
    #iterator = tqdm(items, desc="Drawing Terminal Velocity vs SFR figure") if verbose else items
    for tv, sfr, c in items:
        #plt.errorbar(sfr, tv, xerr=None, yerr=tv_err, color = c, **style)
        sc = ax.scatter(sfr, tv, marker='o', s = 12, color=c, ec='k', lw=.75, alpha=.8)

    x_pos = 0
    y_pos = 0
    x_pos_data, y_pos_data = ax.transData.inverted().transform(ax.transAxes.transform((x_pos, y_pos)))
    ax.errorbar(x_pos_data, y_pos_data, yerr=np.mean(terminal_velocity_errors), fmt='none', color='black', capsize=3, elinewidth=1.5)

    sm = cm.ScalarMappable(cmap=cmap, norm=normalized)
    sm.set_array([])
    #cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    #cbar.set_label(label=r"$R / R_e$", rotation=270, labelpad=21)
    ## setup power law and pearsonr
    if power_law:
        def wind_model(sfr, scale, power):
            return scale * sfr ** power
        
        popt, pcov = curve_fit(wind_model, 10**(sfrs), terminal_velocities, p0=(100, 0.1))
        modsfr = np.logspace(sfrs.min(), 0, 1000)
        modv = wind_model(modsfr, popt[0], popt[1])

        #model_label = rf'$v = {popt[0]:.0f}\ \left( \Sigma_{{\mathrm{{SFR}}}} \right)^{{{popt[1]:.2f}}}$'
        model_label = rf'$v \propto \Sigma_{{\mathrm{{SFR}}}} ^{{{popt[1]:.2f} \pm {np.sqrt(pcov[1,1]):.2f}}}$'
        #model_label = rf'$\alpha = {popt[1]:.2f} \pm {np.sqrt(pcov[1,1]):.2f}$'
        ax.plot(np.log10(modsfr), modv, 'k', linestyle='dashed', 
                label=model_label)

        ax.legend(frameon=False, fontsize=17)

    ax.set_xlim(np.floor(xmin*10)/10,np.ceil(xmax*10)/10)
    ax.set_xlabel(r'$\mathrm{log}\ \Sigma_{\mathrm{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spx^{-1}} \right)$')
    ax.set_ylabel(r'$ v_{\mathrm{out,\ max}}\ \left( \mathrm{km\ s^{-1}} \right)$')

    if save:
        if output_dir is None:
            results_dir = defaults.get_fig_paths(galname, bin_method, subdir='results')
            output_dir = os.path.join(results_dir, 'scatter')
            util.check_filepath(output_dir, verbose=verbose)

        outfil = os.path.join(output_dir, f'{galname}-{bin_method}-terminalv_vs_sfr.pdf')
        plt.savefig(outfil, bbox_inches='tight')
        verbose_print(verbose, f"Terminal velocity vs SFR fig saved to {outfil}")    

    if show:
        plt.show()
    else:
        plt.close()


def plot_galaxy_sample(highlight_gal = None, show=True, save=False, verbose=False):
    from scipy.ndimage import gaussian_filter
    paper_figures_dir = defaults.get_default_path('figures', ensure_exists=True)

    repodir = os.path.abspath(os.path.dirname(defaults.get_default_path()))
    supplemental = os.path.join(repodir, 'data/supplemental')
    mpajhufil = 'galSpecExtra-dr8.fits'
    MADfil = 'MAD_sample.dat'

    mpajhupath = os.path.join(supplemental, mpajhufil)
    MADpath = os.path.join(supplemental, MADfil)

    if not os.path.isfile(mpajhupath):
        raise ValueError(f"FILE NOT FOUND: {mpajhupath}")
    if not os.path.isfile(MADpath):
        raise ValueError(f"FILE NOT FOUND: {MADpath}")
    
    with fits.open(mpajhupath) as hdu:
        mpajhu = hdu[1].data

    mad = ascii.read(MADpath)
    MADlogM = mad['col6']
    MADlogSFR = np.log10(mad['col7'])

    
    sdssind = (mpajhu['LGM_TOT_P50'] >= 7.0) & (mpajhu['SFR_TOT_P50'] > -10.0)
    mpajhu_select = mpajhu[sdssind] #SDSS (Brinchmann et al. 2004)

    ## bin sdss sample from min/max by 0.1 dex
    dex = 0.1
    
    bins = [
        np.arange(np.min(mpajhu_select['LGM_TOT_P50']), np.max(mpajhu_select['LGM_TOT_P50']) + dex, dex), 
        np.arange(np.min(mpajhu_select['SFR_TOT_P50']), np.max(mpajhu_select['SFR_TOT_P50']) + dex, dex)
        ]
    H, xedges, yedges = np.histogram2d(mpajhu_select['LGM_TOT_P50'], mpajhu_select['SFR_TOT_P50'], bins=bins)
    #H = gaussian_filter(H, sigma=.5)
    H = H.T
    H_masked = np.ma.masked_where(H == 0, H)

    H_flat = H.flatten()
    H_sorted = np.sort(H_flat)[::-1]
    cumsum = np.cumsum(H_sorted)
    cumsum /= cumsum[-1]  # normalize to 1

    # Define probability levels corresponding to 1σ, 2σ, 3σ
    sigma_levels = [0.6827, 0.9545, 0.9973] # 0.8664 --> 1.5 sig
    

    # Find the histogram values that correspond to these cumulative probabilities
    H_levels = [H_sorted[np.searchsorted(cumsum, p)] for p in sigma_levels][::-1]

    # Add a top level to close the last filled region
    H_levels.append(H.max() + 1)

    fig, ax = plt.subplots()
    cmap = util.seaborn_palette('gist_gray')
    #levels = [10, 15, 75, 400, H_masked.max() + 1]
    #colors_contour = ["#A39E9E", "#767575", "#414141", '#000000']
    #levels = [7, 15, 50, H_masked.max() + 1]

    colors_contour = ["#A39E9E", "#5F5F5F", '#000000']

    ax.contourf(xedges[:-1], yedges[:-1], H_masked, 
                levels=H_levels, 
                colors=colors_contour,
                antialiased=True, 
                zorder=1,
                alpha=0.9)
    
    #ax.contourf(xedges[:-1], yedges[:-1], H_masked, levels=[10, 50, H_masked.max()], colors=['lightgray','gray','black'], antialiased=True, zorder=5)
    

    ax.scatter(mpajhu_select['LGM_TOT_P50'], mpajhu_select['SFR_TOT_P50'], color='lightgray', s=.5, alpha = .5, zorder=0, rasterized=True)

    ax.scatter(MADlogM, MADlogSFR, marker='D', facecolor='violet', edgecolors='magenta', linewidths=.5, s=10, zorder=10)

    if highlight_gal is not None:
        row = np.argwhere(mad['col1'] == highlight_gal)
        ax.scatter(MADlogM[row], MADlogSFR[row], marker='D', facecolor="#53AFFF", edgecolors="#063CFF", linewidths=1, zorder=11, s=35)# facecolor='violet', edgecolors='magenta', linewidths=.5, s=10, zorder=10)
        ax.scatter(MADlogM[row], MADlogSFR[row], marker='*', color="#E4CA00", linewidths=0.5, zorder=11, s=5)
    ax.set_xlabel(r'$\mathrm{log\ M_{\star} / M_{\odot}}$')
    ax.set_ylabel(r'$\mathrm{log\ SFR\ \left( M_{\odot}\ yr^{-1} \right)}$')

    ax.set_ylim(-3, 3)
    ax.set_xlim(7, 12.5)
    if save:
        figname = 'MAD_logM_logSFR.pdf'
        plt.savefig(os.path.join(paper_figures_dir, figname))
    plt.show() if show else plt.close()



def terminal_velocity_subplots(galname, show = False, save = True, verbose = False):
    paper_figures_dir = defaults.get_default_path('figures', ensure_exists=True)
    bin_methods = ['SQUARE0.6', 'SQUARE2.0']
    datadict = {}

    base_w, base_h = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(1, 2, figsize=(base_w*2, base_h))

    for bin_method, ax in zip(bin_methods, axes):
        datapaths = file_handler.init_datapaths(galname, bin_method)
        with fits.open(datapaths['LOCAL']) as hdul:
            spatial_bins = hdul['SPATIAL_BINS'].data

            sfrmap = hdul['SFRSD'].data
            sfrmap_mask = hdul['SFRSD_MASK'].data
            #sfrmap_error = hdul['SFRSD_ERROR'].data

            vterm = hdul['V_MAX_OUT'].data
            vterm_mask = hdul['V_MAX_OUT_MASK'].data

            vfrac = hdul['v_nai_frac'].data

        datamask = np.logical_or(sfrmap_mask.astype(bool), vterm_mask.astype(bool))
        fracmask = vfrac > -0.95
        combined_mask = datamask | fracmask | (sfrmap == -999) | (vterm >400)

        data = {'sfr':sfrmap, 'vterm':vterm, 'bins':spatial_bins}
        datadict[bin_method] = data

        good_data = util.extract_unique_binned_values(data, spatial_bins, mask = combined_mask)
        xmin, xmax = good_data['sfr'].min(), good_data['sfr'].max()

        ax.scatter(good_data['sfr'], good_data['vterm'], marker='o', s=15, fc = '#0063ff', ec = '#3327e7', alpha=0.75)

        def wind_model(sfr, scale, power):
            return scale * sfr ** power
        
        popt, pcov = curve_fit(wind_model, 10**(good_data['sfr']), good_data['vterm'], p0=(100, 0.1))
        perr = np.sqrt(np.diag(pcov))

        scale, power = popt
        err_scale, err_power = perr
        cov_scalepower = pcov[0,1]
        
        modsfr = np.logspace(xmin - 1, xmax + 1, 1000)
        modv = wind_model(modsfr, scale, power)

        dy_dscale = modsfr ** power
        dy_dpower = scale * modsfr ** power * np.log(modsfr)

        error_v = np.sqrt((dy_dscale * err_scale)**2 + (dy_dpower * err_power)**2 + 2 * cov_scalepower * dy_dscale * dy_dpower)

        model_label = rf'$v \propto \Sigma_{{\mathrm{{SFR}}}} ^{{{popt[1]:.2f} \pm {err_power:.2f}}}$'
        ax.plot(np.log10(modsfr), modv, 'k', linestyle='dashed', 
                label=model_label)
        ax.fill_between(np.log10(modsfr), modv - error_v, modv + error_v, color='#ce0014', alpha=0.3)

        pearson_result = pearsonr(good_data['sfr'], good_data['vterm'])
        val = round(pearson_result[0], 2)
        #ax.legend(frameon=False, fontsize=16, handlelength=0.75)

        ax.set_xlim(np.floor(xmin * 10)/10,np.ceil(xmax * 10)/10)
        

        bin_string = bin_method[-3:]
        ax.text(0.05, 0.95, f"{model_label}"
                            "\n"
                            fr"$r = {val}$", transform = ax.transAxes, va='top', ha='left', fontsize = 14)
        
        ax.text(0.95, 0.925, fr"${bin_string}''$", fontfamily='monospace', transform = ax.transAxes, va='top', ha='right', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='k', alpha=.4))
        ax.set_box_aspect(1)
    
    axes[0].set_ylabel(r'$ v_{\mathrm{out,\ max}}\ \left( \mathrm{km\ s^{-1}} \right)$')
    fig.text(0.5, 0, r'$\mathrm{log}\ \Sigma_{\mathrm{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spx^{-1}} \right)$', va='center', ha='center')
    if save:
        fname = f"{galname}_TERMINALVELOCITY.pdf"
        figout = os.path.join(paper_figures_dir, fname)
        plt.savefig(figout)
        util.verbose_print(verbose, f"V max out subplots plot saved to {figout}")
    if show:
        plt.show()
    else:
        plt.close()
    


def make_paper_plots(galname, highlight_gal = None, show = False, save = True, verbose = False):
    paper_figures_dir = defaults.get_default_path('figures', ensure_exists=True)

    bin_methods = ['SQUARE0.6', 'SQUARE2.0']
    datadict = {}

    alphabet = list(string.ascii_lowercase)

    for bin_method in bin_methods:
        datapaths = file_handler.init_datapaths(galname, bin_method)
        with fits.open(datapaths['LOCAL']) as hdu:
            snrs = hdu['nai_snr'].data
            spatial_bins = hdu['spatial_bins'].data
            vmap = hdu['v_nai'].data
            vmap_mask = hdu['v_nai_mask'].data


        datadict[bin_method] = {'snr':snrs, 'bins':spatial_bins, 'vmap':vmap, 'vmask':vmap_mask}

    
    ### SNR HISTOGRAM
    base_w, base_h = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(1, 2, figsize = (base_w * 2, base_h))
    for i, (bin_method, ax) in enumerate(zip(bin_methods, axes)):
        subdict = datadict[bin_method]
        snrs_all = util.extract_unique_binned_values(subdict['snr'], subdict['bins'])
        snrs_good, snrs_bad = util.extract_unique_binned_values(subdict['snr'], subdict['bins'], subdict['vmask'], return_bad=True)
        inflows = np.logical_and(~subdict['vmask'].astype(bool), vmap > 0)
        outflows = np.logical_and(~subdict['vmask'].astype(bool), vmap < 0)

        histbins = np.linspace(snrs_all.min(), snrs_all.max(), 30)
        ax.hist(snrs_all, bins=histbins, color='k', histtype = 'stepfilled')#, label = 'Full Sample')

        histbins = np.linspace(min(snrs_bad.min(), snrs_good.min()), max(snrs_bad.max(), snrs_good.max()), 30)
        ax.hist(snrs_bad, bins=histbins, color='gray', histtype = 'step', hatch='//', linewidth = 2)#, label=r'low $\mathrm{EW_{Na\ D}}$')
        #ax.hist(snrs_good, bins=histbins, color='k', histtype='stepfilled')#, label='Main sample')

        snrs_inflow = util.extract_unique_binned_values(subdict['snr'], subdict['bins'], ~inflows)
        snrs_outflow = util.extract_unique_binned_values(subdict['snr'], subdict['bins'], ~outflows)

        histbins = np.linspace(min(snrs_inflow.min(), snrs_outflow.min()), max(snrs_inflow.max(), snrs_outflow.max()), 30)
        ax.hist(snrs_outflow, bins=histbins, color='b', histtype = 'step', hatch=r'\\\\', linewidth = 2)#, label='Outflows')
        ax.hist(snrs_inflow, bins=histbins, color='r', histtype = 'step', linewidth = 2)#, label='Inflows')

        strs = ['Full Sample', r'Low $\mathrm{EW_{Na\ D}}$', 'Outflows', 'Inflows']
        colors = ['k', 'gray', 'b', 'r']

        for label, color in zip(strs, colors):
            ax.plot(0,0, label=label, color=color)
        ax.legend(frameon=False, handlelength=0.75, fontsize=14)
        ax.set_box_aspect(1)
        bin_string = bin_method[-3:]
        
        ax.text(0.95, 0.1, fr"${bin_string}''$", fontfamily='monospace', transform = ax.transAxes, va='bottom', ha='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='k', alpha=.5))
        
        ax.text(0.05, 0.95, f"({alphabet[i]})", va='top', ha='left', transform=ax.transAxes, fontsize=16)
    #axes[0].set_xlabel(r'$S/N\ \mathrm{\left( NaI\ 5891,5897 \right)}$')
    axes[0].set_ylabel(r'Number of Bins', fontsize = 20)
    fig.text(0.5, 0, r'$S/N\ \mathrm{\left( NaI\ 5891,5897 \right)}$', ha='center', va='center', fontsize=20)
    if save:
        figname = f"{galname}_SNR_HISTS.pdf"
        plt.savefig(os.path.join(paper_figures_dir, figname))
    plt.show() if show else plt.close()

    ### GALAXY SAMPLE
    plot_galaxy_sample(show=show, save=save, verbose=verbose)
    terminal_velocity_subplots(galname, show=show, save=save, verbose=verbose)

def get_args():
    parser = argparse.ArgumentParser(description="A script to create/overwrite plots without rerunning analyze_NaI")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)
    parser.add_argument('--paper', help = "Make plots exclusively for the manuscript (default: False)", action='store_true', default = False)

    return parser.parse_args()


def make_plots(galname, bin_method, paper = False, verbose = False):
    warnings.filterwarnings('ignore')

    mplconfig = os.path.join(defaults.get_default_path('config'), 'figures.mplstyle')
    plt.style.use(mplconfig)

    verbose_print(verbose, f"Creating Plots for {galname}-{bin_method}")
    
    plotter.DAP_MAP_grid(galname, bin_method, verbose=verbose)
    plotter.local_MAP_grid(galname, bin_method, mask=True, verbose=verbose)
    plot_local_maps(galname, bin_method, verbose=verbose)
    velocity_vs_sfr(galname, bin_method, pearson=True, contours=True, hists=True, verbose=verbose)
    terminal_velocity(galname, bin_method, power_law=True, verbose=verbose)
    plotter.gas_properties_scatter(galname, bin_method, pearson=True, verbose=verbose)
    plotter.incidence(galname, bin_method, verbose=verbose)

    if paper:
        make_paper_plots(galname, verbose=verbose)
        #plotter.plot_bin_profiles(galname, bin_method, verbose=verbose)
        #plotter.bin_profiles(galname, bin_method, show_vmax_out=False)
        plotter.enhanced_corner_plotter(galname, bin_method, [950], verbose=verbose)

    verbose_print(verbose, 'Done.')

def main():
    args = get_args()

    galname = args.galname
    bin_method = args.bin_method
    verbose = args.verbose
    paper = args.paper
    make_plots(galname, bin_method, paper, verbose)

if __name__ == "__main__":
    main()