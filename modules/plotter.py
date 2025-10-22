import os
import numpy as np
from astropy.io import fits
from astropy.table import Table

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from modules import defaults, util, file_handler, inspect, model_nai
from mcmc_results import sort_paths

import string
import corner
import re

import seaborn as sns
import pandas as pd
import cmasher as cmr

from typing import Optional

plt.style.use(defaults.matplotlib_rc())


def DAP_MAP_grid(galname: str, bin_method: str, show = False, save = True, verbose = False):
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    mapsfile = datapath_dict['MAPS']

    def rescale_8bit(image, cmin = 0, cmax = None, scale = 'linear'):
        cmax = image.max() if cmax is None else cmax

        rescale = (image - cmin) / (cmax - cmin)
        if scale == 'linear':
            scaledim = 255 * rescale
        elif scale == 'sqrt':
            scaledim = 255 * np.sqrt( rescale )
            
        scaledim[scaledim < 0] =0
        scaledim[scaledim >255] = 255
        return scaledim.astype(np.uint8)
    

    def rgb_im(cubefile, threshold = 100):
        from mpdaf.obj import Cube
        cube = Cube(cubefile)
        imB = cube.get_band_image('Johnson_B').data
        imV = cube.get_band_image('Johnson_V').data
        imR = cube.get_band_image('Cousins_R').data

        B = rescale_8bit(imB, cmax=900, scale='sqrt')
        V = rescale_8bit(imV, cmax=1000, scale='sqrt')
        R = rescale_8bit(imR, cmax=750, scale='sqrt')

        ny,nx = imB.shape
        rgb = np.zeros([ny,nx,3], dtype=np.uint8)
        rgb[:,:,0] = R
        rgb[:,:,1] = V
        rgb[:,:,2] = B

        # Case 1: Only one channel is 255, and the other two are low
        r_peak = (rgb[..., 0] == 255) & (rgb[..., 1] < threshold) & (rgb[..., 2] < threshold)
        g_peak = (rgb[..., 1] == 255) & (rgb[..., 0] < threshold) & (rgb[..., 2] < threshold)
        b_peak = (rgb[..., 2] == 255) & (rgb[..., 0] < threshold) & (rgb[..., 1] < threshold)

        # Case 2: Two channels are 255, and one is low
        rg_peak = (rgb[..., 0] == 255) & (rgb[..., 1] == 255) & (rgb[..., 2] < threshold) 
        rb_peak = (rgb[..., 0] == 255) & (rgb[..., 2] == 255) & (rgb[..., 1] < threshold) 
        gb_peak = (rgb[..., 1] == 255) & (rgb[..., 2] == 255) & (rgb[..., 0] < threshold)  

        mask = r_peak | g_peak | b_peak | rg_peak | rb_peak | gb_peak

        rgb[mask] = 0
        return rgb

    with fits.open(mapsfile) as maps:
        snr = maps['bin_snr'].data
        radius = maps['bin_lwellcoo'].data[1]

        chisq = maps['stellar_fom'].data[2]
        stellar_vel = maps['stellar_vel'].data
        stellar_vel_mask = util.spec_mask_handler(maps['stellar_vel_mask'].data).astype(bool)

        stellar_sigma = maps['stellar_sigma'].data
        stellar_sigma_mask = util.spec_mask_handler(maps['stellar_sigma_mask'].data).astype(bool)

        emlines = maps['EMLINE_GFLUX'].data
        emline_mask = util.spec_mask_handler(maps['EMLINE_GFLUX_mask'].data).astype(bool)


    pipepline_datapath = defaults.get_data_path('pipeline')
    muse_cubes_path = os.path.join(pipepline_datapath, 'muse_cubes')
    raw_cube_path = os.path.join(muse_cubes_path, galname, f"{galname}.fits")

    rgb = rgb_im(raw_cube_path)

    ha = emlines[23]
    ha_mask = emline_mask[23]
    hb = emlines[14]
    hb_mask = emline_mask[14]

    plotdicts = {
        'RGB':dict(image = rgb, mask = None, cmap = None, vmin = None, vmax = None, v_str = r'$B, V, R$'),
        'RADIUS':dict(image = radius, mask = None, cmap = util.seaborn_palette('autumn_r'), vmin=0, vmax=1, v_str = r'$R / R_e$'),
        'SNR':dict(image = snr, mask = None, cmap = util.seaborn_palette('mako'), vmin=0, vmax=75, v_str = r'$S/N_g$'),
        'CHISQ':dict(image = chisq, mask = None, cmap = util.seaborn_palette('binary_r'), vmin = 0, vmax = 2, v_str = r'$\chi^2_{\nu}$'),

        'STELLAR_VEL':dict(image = stellar_vel, mask = stellar_vel_mask, cmap = util.seaborn_palette('seismic'), vmin = -250, vmax = 250, v_str = r'$V_{\star}\ \left( \mathrm{km\ s^{-1}} \right)$'),
        'STELLAR_SIG':dict(image = stellar_sigma, mask =stellar_sigma_mask, cmap = cmr.ember, vmin = 25, vmax = 100, v_str = r'$\sigma_{\star}\ \left( \mathrm{km\ s^{-1}} \right)$'),
        'H_alpha':dict(image = ha, mask = ha_mask, cmap = util.seaborn_palette('bone'), vmin = 0, vmax = 10, v_str = r'$F_{\mathrm{H}\alpha}$'), # \left( \mathrm{10^{-17}\ erg\ s^{-1}\ cm^{-2}\ spaxel^{-1}} \right)
        'H_beta':dict(image = hb, mask = hb_mask, cmap = util.seaborn_palette('bone'), vmin = 0, vmax = 2, v_str = r'$F_{\mathrm{H}\beta}$'), # \left( \mathrm{10^{-17}\ erg\ s^{-1}\ cm^{-2}\ spaxel^{-1}} \right)
    }
    
    alphabet = list(string.ascii_lowercase)

    nrow = 2
    ncol = 4

    # Setup figure and GridSpec
    base_w, base_h = plt.rcParams["figure.figsize"]
    fig = plt.figure(figsize=(ncol*base_w, nrow*base_h))
    gs = GridSpec(nrow, ncol, figure=fig, hspace=.375, wspace=-.35)

    # Create the 3x3 axes
    axes = []
    for i in range(nrow):
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            ax.set_aspect('equal')  # Keep square axes
            axes.append(ax)

    # Iterate over the axes and plot content
    for a, key, plot_dict, char in zip(axes, plotdicts.keys(), plotdicts.values(), alphabet):
        plotmap = plot_dict['image']
        plotmask = plot_dict['mask']
        if key != "RGB":
            plotmap[plotmap == 0] = np.nan
            if plotmask is not None:
                plotmap[plotmask] = np.nan
        im = a.imshow(plotmap, origin='lower',
                    vmin=plot_dict['vmin'], vmax=plot_dict['vmax'],
                    cmap=plot_dict['cmap'], extent=[32.4, -32.6, -32.4, 32.6])

        a.set_facecolor('lightgray')

        # Colorbar
        divider = make_axes_locatable(a)
        cax = divider.append_axes("top", size="5%", pad=0.01)
        value_string = plot_dict['v_str']
        if key == 'RGB':
            dummy_data = np.zeros(plotmap.shape[:2])
            dummy_cmap = mcolors.ListedColormap(['none'])
            dummy_norm = mcolors.Normalize(vmin=0, vmax=1)
            cbar = fig.colorbar(plt.imshow(dummy_data, cmap=dummy_cmap, norm=dummy_norm),
                                cax=cax, orientation='horizontal')
            cbar.ax.set_facecolor('white')
            cbar.set_ticks([])
            cbar.set_label(value_string, labelpad=-45)
            cbar.outline.set_visible(False)
            cbar.ax.patch.set_alpha(0)
        else:
            def smart_int_formatter(x, pos):
                if abs(x - round(x)) < 1e-2:  # very close to integer
                    return f'{int(round(x))}'
                else:
                    return ''  # empty string means no label
                
            cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
            cbar.set_label(value_string, labelpad=-45)
            cax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_major_formatter(FuncFormatter(smart_int_formatter))


        a.text(0.075, 0.9, f'({char})', fontsize=10, transform=a.transAxes, color='white')

    for i, ax in enumerate(axes):
        row, col = divmod(i, ncol)
        if row < nrow-1:  # Hide x ticks except for bottom row
            ax.set_xticklabels([])
        if col > 0:  # Hide y ticks except for left column
            ax.set_yticklabels([])

    # Axis labels for the whole figure
    fig.text(0.5, 0.025, r'$\Delta \alpha$ (arcsec)', ha='center', va='center', fontsize = 18)
    fig.text(0.12, 0.5, r'$\Delta \delta$ (arcsec)', ha='center', va='center', rotation='vertical', fontsize = 18) 

    if save:
        results_dir = defaults.get_fig_paths(galname, bin_method, subdir='results')
        output_dir = os.path.join(results_dir, 'maps')
        outfile = os.path.join(output_dir, f"{galname}-{bin_method}_DAPmaps.pdf")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(outfile)
        util.verbose_print(verbose, f"DAP map grid saved to: {outfile}")
    if show:
        plt.show()
    else:
        plt.close()



def local_MAP_grid(galname: str, bin_method: str, mask = True, show = False, save = True, verbose = False):
    util.verbose_print(verbose, "Creating grid plot...")
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    local_maps_path = datapath_dict['LOCAL']
    
    with fits.open(local_maps_path) as hdul:
        snr = hdul['nai_snr'].data
        ew = hdul['ew_noem'].data
        sfrsd = hdul['sfrsd'].data
        v_bulk = hdul['v_nai'].data

        if mask:
            #snr_mask = hdul['nai_snr_mask'].data.astype(bool)
            snr_mask = snr <= 0
            ew_mask = hdul['ew_noem_mask'].data.astype(bool)
            sfrsd_mask = hdul['sfrsd_mask'].data.astype(bool)
            v_mask = hdul['v_nai_mask'].data.astype(bool)
        else:
            snr_mask = np.zeros_like(snr).astype(bool)
            ew_mask = np.zeros_like(ew).astype(bool)
            sfrsd_mask = np.zeros_like(sfrsd).astype(bool)
            v_mask = np.zeros_like(v_mask).astype(bool)


    plotdicts = {
        'SNR':dict(image = snr, mask = snr_mask, cmap = cmr.sapphire, facecolor = 'k', vmin = 0, vmax = 100, v_str = r'$S/N_{\mathrm{Na\ D}}$'),
        'EW':dict(image = ew, mask = ew_mask, cmap = cmr.gem, facecolor = 'k', vmin=-0.5, vmax=2, v_str = r'$\mathrm{EW_{Na\ D}}\ \left( \mathrm{\AA} \right)$'),

        'SFRSD':dict(image = sfrsd, mask = sfrsd_mask, cmap = util.seaborn_palette('rainbow'), facecolor = 'lightgray', vmin=-2.5, vmax=0, v_str = r'$\mathrm{log\ \Sigma_{SFR}}\ \left( \mathrm{M_{\odot}\ kpc^{-2}\ yr^{-1}} \right)$'),

        'V_BULK':dict(image = v_bulk, mask = v_mask, cmap = util.seaborn_palette('seismic'), facecolor = 'lightgray', vmin = -200, vmax = 200, v_str = r'$v_{\mathrm{cen}}\ \left( \mathrm{km\ s^{-1}} \right)$'),
    }
    
    alphabet = list(string.ascii_lowercase)

    nrow = 1
    ncol = 4

    # Setup figure and GridSpec
    base_w, base_h = plt.rcParams["figure.figsize"]
    fig = plt.figure(figsize=(ncol*base_w, nrow*base_h))
    gs = GridSpec(nrow, ncol, figure=fig, hspace=0, wspace=0.1)

    # Create the nrow x ncol axes
    axes = []
    for i in range(nrow):
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            ax.set_aspect('equal')  # Keep square axes
            axes.append(ax)

    # Iterate over the axes and plot content
    for a, key, plot_dict, char in zip(axes, plotdicts.keys(), plotdicts.values(), alphabet):
        plotmap = plot_dict['image']
        plotmask = plot_dict['mask']

        plotmap[plotmask] = np.nan
        # TODO mask???
        im = a.imshow(plotmap, origin='lower',
                    vmin=plot_dict['vmin'], vmax=plot_dict['vmax'],
                    cmap=plot_dict['cmap'], extent=[32.4, -32.6, -32.4, 32.6])

        a.set_facecolor(plot_dict['facecolor'])

        # Colorbar
        divider = make_axes_locatable(a)
        cax = divider.append_axes("top", size="5%", pad=0.01)
        value_string = plot_dict['v_str']

        def smart_int_formatter(x, pos):
            if abs(x - round(x)) < 1e-2:  # very close to integer
                return f'{int(round(x))}'
            else:
                return ''  # empty string means no label
            
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label(value_string, labelpad=-45)
        cax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(smart_int_formatter))


        a.text(0.075, 0.9, f'({char})', fontsize=10, transform=a.transAxes, color='white')

    for i, ax in enumerate(axes):
        if i > 0:
            ax.set_yticklabels([])

    # Axis labels for the whole figure
    fig.text(0.5, 0.0, r'$\Delta \alpha$ (arcsec)', ha='center', va='center', fontsize=18)
    fig.text(0.08, 0.5, r'$\Delta \delta$ (arcsec)', ha='center', va='center', rotation='vertical', fontsize=18)
    # Save output
    if save:
        results_dir = defaults.get_fig_paths(galname, bin_method, subdir='results')
        output_dir = os.path.join(results_dir, 'maps')
        outfile = os.path.join(output_dir, f"{galname}-{bin_method}_MAPgrid.pdf")
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(outfile)
        util.verbose_print(verbose, f"Results map grid saved to: {outfile}")
    if show:
        plt.show()
    else:
        plt.close()



def MAP_plotter(data: np.ndarray, cbar_label: str, directory: str, figname: str,
                mask_data: Optional[np.ndarray] = None, verbose = False, **imshow_kwargs):

    plotmap = np.copy(data)
    if mask_data is not None:
        plotmap[mask_data.astype(bool)] = np.nan

    plt.figure()
    im = plt.imshow(plotmap, origin='lower', extent=[32.4, -32.6,-32.4, 32.6], **imshow_kwargs)
    plt.xlabel(r'$\Delta \alpha$ (arcsec)')
    plt.ylabel(r'$\Delta \delta$ (arcsec)')
    ax = plt.gca()
    ax.set_facecolor('lightgray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.01)

    cbar = plt.colorbar(im, cax=cax, orientation = 'horizontal')
    cbar.set_label(f"{cbar_label}", labelpad=-55)
    cax.xaxis.set_ticks_position('top')
    
    util.check_filepath(directory, verbose=verbose)
    savepath = os.path.join(directory, figname)
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    util.verbose_print(verbose, f'MAP "{figname}" saved to "{savepath}"')



def HIST_plotter(data: np.ndarray, spatial_bins: np.ndarray, x_label: str, directory: str, figname: str,
                mask_data: Optional[np.ndarray] = None, show_masked = True, verbose = False, **hist_kwargs):

    ## separate good bins/data and bad bins/data using the mask
    if mask_data is not None:
        good_data, bad_data = util.extract_unique_binned_values(data, spatial_bins, mask_data, return_bad=True)
        bad_data = bad_data[bad_data!=-999] # get rid of default -999 values
        histbins = np.linspace(min(bad_data.min(), good_data.min()), max(bad_data.max(), good_data.max()), 40)
    else:
        good_data = util.extract_unique_binned_values(data, spatial_bins)
        histbins = np.linspace(good_data.min(),  good_data.max(), 40)

    plt.figure()
    plt.hist(good_data, bins=histbins, color='k', **hist_kwargs)
    
    if mask_data is not None and show_masked:
        plt.hist(bad_data, bins=histbins, edgecolor='r', hatch='//', fill=False, linewidth=1)

    plt.xlabel(rf"{x_label}")
    plt.ylabel(r"$N_{\mathrm{bins}}$")

    util.check_filepath(directory, verbose=verbose)
    save_path = os.path.join(directory, figname)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    util.verbose_print(verbose, f'HISTOGRAM "{figname}" saved to "{save_path}"')



def custom_corner_plot(samples, labels, units, units_closed, truths=None):
    """
    Simplified custom corner plot that avoids contour level issues
    """
    n_params = samples.shape[1]
    
    base_w, base_h = plt.rcParams["figure.figsize"]
    fig = plt.figure(figsize=(n_params*base_w, n_params*base_h))
    
    gs = fig.add_gridspec(n_params, n_params, hspace=0, wspace=0)
    
    axes = np.empty((n_params, n_params), dtype=object)
    
    xlims = [(samples[:, i].min(), samples[:, i].max()) for i in range(n_params)]

    # Create axes with proper sharing
    for i in range(n_params):
        for j in range(n_params):
            if i >= j:
                axes[i, j] = fig.add_subplot(gs[i, j])


    for i in range(n_params):
        for j in range(n_params):
            if i<j:
                continue

            ax = axes[i, j]
            
            if i == j:
                # 1D histogram
                ax.hist(samples[:, i], bins=30, density=True, alpha=0.7, 
                       histtype = 'step', edgecolor='black')
                
                # Add percentiles
                p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
                delp84 = p84 - p50
                delp16 = p50 - p16

                ax.set_title(rf"${labels[i]} = {p50:.2f}^{{+{delp84:.2f}}}_{{-{delp16:.2f}}}\ \mathrm{{{units[i]}}}$")
                ax.tick_params(axis='y', which='both', length=0)
                #ax.set_xlim(samples[:,i].min(), samples[:,i].max())
                # ax.set_xlim(xlims[i])
                # ax.set_ylim(bottom=0)

                ax.axvline(p16, color='k', linestyle='dotted', alpha=0.8)
                ax.axvline(p50, color='k', linestyle='dashed', alpha=0.8)
                ax.axvline(p84, color='k', linestyle='dotted', alpha=0.8)
                
                if truths is not None:
                    ax.axvline(truths[i], color='k', linestyle='dashed', linewidth=2)
                
                if i == n_params - 1:
                    ax.set_xlabel(fr"${labels[j]}\ {units_closed[j]}$")
                else:
                    ax.set_xlabel('')
                
            elif i > j:
                samples_fixed = samples.astype(np.float64, copy=False)

                temp_df = pd.DataFrame({
                    labels[j]: samples_fixed[:, j],
                    labels[i]: samples_fixed[:, i]
                })
                                         
                # Plot with seaborn
                sns.scatterplot(data=temp_df, x=labels[j], y=labels[i], 
                            ax=ax, alpha=.95, s=5, color='dimgray')
                sns.kdeplot(data=temp_df, x=labels[j], y=labels[i], 
                        ax=ax, levels=[.3, .6], alpha=.9, color='lightgray', cut=2)
                sns.histplot(data=temp_df, x = labels[j], y = labels[i], bins=60, ax=ax, pthresh=0.075, cmap='mako')

                if truths is not None:
                    ax.scatter(truths[j], truths[i], s=100, c='orange',
                            marker='*', zorder=10)
                
                if i == n_params - 1:
                    ax.set_xlabel(fr"${labels[j]}\ {units_closed[j]}$")
                else:
                    ax.set_xlabel('')

                if j == 0:
                    ax.set_ylabel(fr"${labels[i]}\ {units_closed[i]}$")
                else:
                    ax.set_ylabel('')

                # ax.set_xlim(samples[:,j].min(), samples[:,j].max())
                # ax.set_ylim(samples[:,i].min(), samples[:,i].max())
                # ax.set_xlim(xlims[j])
                # ax.set_ylim(xlims[i])
            
            #ax.set_box_aspect(1)
    for row in range(n_params):
        for col in range(n_params):
            if axes[row, col] is not None:
                ax = axes[row,col]
                if row != n_params - 1:
                    ax.set_xticklabels([])
                if row == 0 or col > 0 :
                    ax.set_yticklabels([])

    # Set all limits and aspects AFTER all plotting is done
    for i in range(n_params):
        for j in range(n_params):
            if i >= j:
                ax = axes[i, j]
                ax.set_xlim(xlims[j if i > j else i])
                if i > j:
                    ax.set_ylim(xlims[i])
                #ax.set_aspect('equal', adjustable='box')
    
    return fig, axes


def enhanced_corner_plotter(galname, bin_method, bin_list, show=False, save=True, verbose=False):

    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    mcmc_files = datapath_dict['MCMC']

    with fits.open(datapath_dict['LOGCUBE']) as logcube:
        flux_cube = logcube['FLUX'].data
        ivar_cube = logcube['IVAR'].data
        model_cube = logcube['MODEL'].data
        wavelength = logcube['WAVE'].data
    
    with fits.open(datapath_dict['LOCAL']) as local:
        redshift = local['REDSHIFT'].data
        binmap = local['SPATIAL_BINS'].data

    sorted_paths = sort_paths(mcmc_files)

    NaD_window = (5875, 5915)

    for binID in bin_list:
        util.verbose_print(verbose, f"Obtaining samples for bin {binID}")
        w = binID == binmap
        ny,nx = np.where(w)
        y,x = ny[0], nx[0]

        z = redshift[y,x]
        flux_bin = flux_cube[:,y,x]
        ivar_bin = ivar_cube[:,y,x]
        stellar_flux_bin = model_cube[:,y,x]
        restwave_bin = wavelength/(1+z)

        NaD_lims = (restwave_bin>=NaD_window[0]) & (restwave_bin<=NaD_window[1])

        flux = flux_bin[NaD_lims]
        ivar = ivar_bin[NaD_lims]
        stellar_flux = stellar_flux_bin[NaD_lims]
        restwave = restwave_bin[NaD_lims]

        normflux = flux / stellar_flux
        normerror = (1 / np.sqrt(ivar)) / stellar_flux

        for mcmc_fil in sorted_paths:
            path, file = os.path.split(mcmc_fil)
            match = re.search(r'binid-(\d+)-(\d+)-samples', file)

            if match:
                start_ID = int(match.group(1))
                end_ID = int(match.group(2))

                if start_ID <= binID <= end_ID:
                    util.verbose_print(verbose, f"    File found {file}")
                    data = fits.open(mcmc_fil)
                    data_table = Table(data[1].data)

                    all_bins = data_table['bin'].data

                    i = np.where(binID == all_bins)[0][0]
                    samples = data_table[i]['samples']
                    percentiles = data_table[i]['percentiles']
                    theta = percentiles[:,0]
                    model_dict = model_nai.model_NaI(theta, z, restwave)

                    flat_samples = samples[:,1000:,:].reshape(-1, 4)


                    labels = [r'\lambda_0', r'\log N', r'b_D', r'C_f']
                    units = [r'\AA', r'cm^{-2}', r'km\ s^{-1}', r'']
                    units_closed = [fr'\left( \mathrm{{{u}}} \right)' if u != '' else '' for u in units]

                    # Create the custom corner plot
                    fig_corner, axes = custom_corner_plot(flat_samples, labels, units, units_closed, truths=theta)
                    
                    bbox = axes[0,0].get_position()
                    bbox2 = axes[3,3].get_position()
                    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
                    width = x1 - x0
                    height = y1 - y0

                    left = bbox2.x0

                    dx  = np.ceil((x1 - x0) * 10) / 10
                    dy = np.ceil((y1 - y0) * 10) / 10

                    # Add spectrum plot as an inset or separate subplot
                    ax_spec = fig_corner.add_axes([left, y0 + height/3, width, height * 2/3]) # [left, bottom, width, height]
                    ax_observed = fig_corner.add_axes([left, y0, width, height/3])

                    
                    # Plot spectrum
                    ax_spec.plot(restwave, normflux, 'k-', alpha=1, linewidth=2, drawstyle='steps-mid')
                    #ax_spec.plot(restwave, normerror, 'dimgray', linewidth = 1)
                    ax_spec.plot(model_dict['modwv'], model_dict['modflx'], color='dimgray', 
                                linewidth=4)
                    ax_spec.plot(model_dict['modwv'], model_dict['modflx'], color="#02a5d2", 
                                linewidth=2.75, label='Best Fit')
                    ax_spec.set_ylabel('Normalized Flux')
                    ax_spec.set_xticklabels([])
                    ax_spec.set_xlim(NaD_window)

                    #ax_spec.grid(True, alpha=0.3)
                    #ax_spec.set_ylim(0,1.5)
                    ax_observed.plot(restwave, flux, color='#0b0405', drawstyle = 'steps-mid')
                    ax_observed.plot(restwave, stellar_flux, color="#0063ff", drawstyle='steps-mid')
                    ax_observed.set_xlabel(r'Wavelength $\left( \mathrm{\AA} \right)$')
                    ax_observed.set_xlim(NaD_window)
                    ax_observed.set_ylabel(r'Flux')

                    
                    #plt.tight_layout()
                    
                    if save:
                        output_dir = defaults.get_fig_paths(galname, bin_method, subdir='inspection')
                        fname = f"{galname}-{bin_method}_Enhanced_corner_bin_{binID}.pdf"
                        figpath = os.path.join(output_dir, fname)
                        plt.savefig(figpath, bbox_inches='tight')
                        util.verbose_print(verbose, f"Figure saved to {figpath}\n")
                        if not show:
                            plt.close()
                    
                    if show:
                        plt.show()


def gas_properties_scatter(galname, bin_method, show = False, save = True, verbose = False):
    datapath_dict = file_handler.init_datapaths(galname, bin_method)

    with fits.open(datapath_dict['LOCAL']) as local_hdu:
        spatial_bins = local_hdu['spatial_bins'].data

        vmap = local_hdu['v_nai'].data
        vmap_mask = local_hdu['v_nai_mask'].data
        vfrac = local_hdu['v_nai_frac'].data

        eqw = local_hdu['ew_noem'].data
        eqw_mask = local_hdu['ew_noem_mask'].data

        mcmc_cube = local_hdu['mcmc_results'].data

        logN = mcmc_cube[1]
        bd = mcmc_cube[2]
        Cf = mcmc_cube[3]

        sfrmap = local_hdu['sfrsd'].data
        sfr_mask = local_hdu['sfrsd_mask'].data
        ebv = local_hdu['e(b-v)'].data

    datamask = (vmap_mask + eqw_mask + sfr_mask).astype(bool)
    mcmc_mask = (logN == 0) | (bd == 0) | (Cf == 0)
    outflow_mask = (vmap > 0) | (vfrac < .95)
    inflow_mask = (vmap < 0) | (vfrac < .95)
    mask_out = (datamask + mcmc_mask + outflow_mask).astype(bool)
    mask_in = (datamask + mcmc_mask + inflow_mask).astype(bool)

    labels = [r'$\mathrm{EW_{Na\ D}\ \left( \AA \right)}$', r'$v_{\mathrm{cen}}\ \left( \mathrm{km\ s^{-1}} \right)$']
    ylims = [(-.2,2.5), (-10,210)]


    mapdict = {'EW':eqw, 'velocity':vmap}

    secondary_mapdict = {'extinction':ebv, 'sfr':sfrmap}

    outflow_dict = util.extract_unique_binned_values(mapdict, spatial_bins, mask=mask_out)
    outflow_secondary = util.extract_unique_binned_values(secondary_mapdict, spatial_bins, mask=mask_out)
    inflow_dict = util.extract_unique_binned_values(mapdict, spatial_bins, mask=mask_in)
    inflow_secondary = util.extract_unique_binned_values(secondary_mapdict, spatial_bins, mask=mask_in)

    outflow_dict['velocity'] = abs(outflow_dict['velocity'])


    base_w, base_h = plt.rcParams["figure.figsize"]
    nrow = 2
    ncol = 4

    sub_nrow = 2
    sub_ncol = 2
    fig = plt.figure(figsize = (base_w * ncol, base_h * nrow))

    main_gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05)

    gs1 = main_gs[0,0].subgridspec(sub_nrow, sub_ncol, hspace = 0, wspace = 0.05)
    gs2 = main_gs[0,1].subgridspec(sub_nrow, sub_ncol, hspace = 0, wspace = 0.05)


    gs_list = [gs1, gs2]

    axes = {}
    superlabels = [r"$\mathrm{log}\ \Sigma_{\mathrm{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spaxel^{-1}} \right)$", r"$E(B-V)$"]
    for i, (gs, label) in enumerate(zip(gs_list, superlabels)):
        ax_group = fig.add_subplot(gs[:,:])
        ax_group.set_xlabel(f"{label}", fontsize=22)
        ax_group.tick_params(which = 'both', labelcolor='none', top=False, bottom=False, 
                            left=False, right=False)
        ax_group.spines['top'].set_visible(False)
        ax_group.spines['right'].set_visible(False)
        ax_group.spines['bottom'].set_visible(False)
        ax_group.spines['left'].set_visible(False)

        for row in range(sub_nrow):
            for col in range(sub_ncol):
                ax = fig.add_subplot(gs[row, col])
                icol = i*2+col
                key = f"{row}, {icol}"
                axes[key] = ax

    for irow, key in zip(range(nrow), outflow_dict.keys()):

        label = labels[irow]
        ylim = ylims[irow]

        #kwargs = dict(s=15, marker='o', linewidths=0.75, alpha=1)

        ax1, ax2, ax3, ax4 = [axes[f"{irow}, {i}"] for i in range(ncol)]

        #ax1.scatter(outflow_secondary['sfr'], outflow_dict[key], facecolors = facecolors_b, edgecolors = edgecolor_b, **kwargs)
        ax1.scatter(outflow_secondary['sfr'], outflow_dict[key], c='dimgray', s=1, alpha=0.9)
        util.seaborn_histplot({'sfr':outflow_secondary['sfr']}, {key:outflow_dict[key]}, ax1, cmap=cmr.arctic)
        ax1.set_ylabel(label, fontsize=22)
        ax1.set_xlim(-2.6, 0)
        ax1.set_ylim(ylim)

        ax2.scatter(inflow_secondary['sfr'], inflow_dict[key], c='dimgray', s=1, alpha=0.9)
        util.seaborn_histplot({'sfr':inflow_secondary['sfr']}, {key:inflow_dict[key]}, ax2, cmap=cmr.sunburst)
        ax2.set_ylabel('')
        ax2.set_yticklabels([])
        ax2.set_xlim(-2.6, 0)
        ax2.set_ylim(ylim)

        ax3.scatter(outflow_secondary['extinction'], outflow_dict[key], c='dimgray', s=1, alpha=0.9)
        util.seaborn_histplot({'extinction':outflow_secondary['extinction']}, {key:outflow_dict[key]}, ax3, cmap=cmr.arctic)
        ax3.set_ylabel('')
        ax3.set_yticklabels([])
        ax3.set_xlim(0, 1)
        ax3.set_ylim(ylim)

        ax4.scatter(inflow_secondary['extinction'], inflow_dict[key], c='dimgray', s=1, alpha=0.9)
        util.seaborn_histplot({'extinction':inflow_secondary['extinction']}, {key:inflow_dict[key]}, ax4, cmap=cmr.sunburst)
        ax4.set_ylabel('')
        ax4.set_yticklabels([])
        ax4.set_xlim(0, 1)
        ax4.set_ylim(ylim)


        if irow != nrow - 1:
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
            ax3.set_xticklabels([])
            ax4.set_xticklabels([])
        else:

            ax3.set_xticklabels([0, None, 0.5, None, 1])
            ax4.set_xticklabels([0, None, 0.5, None, 1])

        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax3.set_xlabel('')
        ax4.set_xlabel('')

    for ax in list(axes.values()):
        ax.set_box_aspect(1)
        ax.grid(visible=True, linestyle = 'dotted', alpha=0.5)

    if save:
        results_dir = defaults.get_fig_paths(galname, bin_method, subdir='results')
        scatter_dir = os.path.join(results_dir, 'scatter')
        util.check_filepath(scatter_dir, mkdir=True, verbose=verbose)

        plt.savefig(os.path.join(scatter_dir, f'{galname}-{bin_method}_Vcen_vs_SFR_DUST.pdf'), bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return 
    mcmc_labels = [r'$\mathrm{log}\ N\ \left( \mathrm{cm^{-2}} \right)$', r'$b_D\ \left( \mathrm{km\ s^{-1}} \right)$',r'$C_f$']
    mcmc_ylims = (12.1, 16.4), (-5,110), (0,.95)
    mcmcdict = {'logN':logN, 'bD':bd, 'Cf':Cf}
    outflow_dict = util.extract_unique_binned_values(mcmcdict, spatial_bins, mask=mask_out)
    inflow_dict = util.extract_unique_binned_values(mcmcdict, spatial_bins, mask=mask_in)

    nrow = len(mcmc_labels)
    ncol = 4

    base_w, base_h = plt.rcParams["figure.figsize"]
    fig = plt.figure(figsize=(ncol*base_w, nrow*base_h))
    gs = GridSpec(nrow, ncol, figure=fig, hspace=0, wspace=0.3)

    # Create the nrow x ncol axes
    axes = []
    for i in range(nrow):
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    for irow, key in zip(range(nrow), mcmcdict.keys()):
        label = mcmc_labels[irow]
        ylim = mcmc_ylims[irow]

        #kwargs = dict(s=15, marker='o', linewidths=0.75, alpha=1)

        col_inds = slice(irow * ncol , irow * ncol + ncol)
        ax1, ax2, ax3, ax4 = axes[col_inds]

        ax1.scatter(outflow_secondary['sfr'], outflow_dict[key], c='k', s=1, alpha=0.8)
        ax1.set_ylabel(label)
        ax1.set_xlim(-2.6, 0)
        ax1.set_ylim(ylim)

        ax2.scatter(outflow_secondary['extinction'], outflow_dict[key], c='k', s=1, alpha=0.8)
        ax2.set_yticklabels([])
        ax2.set_xlim(0, 1)
        ax2.set_ylim(ylim)


        ax3.scatter(inflow_secondary['sfr'], inflow_dict[key], c='k', s=1, alpha=0.8)
        #ax3.set_ylabel(label)
        ax3.set_xlim(-2.6, 0)
        ax3.set_ylim(ylim)

        ax4.scatter(inflow_secondary['extinction'], inflow_dict[key], c='k', s=1, alpha=0.8)
        ax4.set_yticklabels([])
        ax4.set_xlim(0, 1)
        ax4.set_ylim(ylim)


        if irow == nrow - 1:
            #ax1.set_xlabel(r"$\mathrm{log}\ \Sigma_{\mathrm{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spaxel^{-1}} \right)$")
            ax1.set_xlabel(r"$\mathrm{log}\ \Sigma_{\mathrm{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}} \right)$")
            ax2.set_xlabel(r"$E(B-V)$")
            #ax3.set_xlabel(r"$\mathrm{log}\ \Sigma_{\mathrm{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spaxel^{-1}} \right)$")
            ax3.set_xlabel(r"$\mathrm{log}\ \Sigma_{\mathrm{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}} \right)$")
            ax4.set_xlabel(r"$E(B-V)$")
        else:
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
            ax3.set_xticklabels([])
            ax4.set_xticklabels([])

            ax1.set_xlabel('')
            ax2.set_xlabel('')
            ax3.set_xlabel('')
            ax4.set_xlabel('')

    # for ax in axes:
    #     ax.set_box_aspect(1)

    if save:
        results_dir = defaults.get_fig_paths(galname, bin_method, subdir='results')
        scatter_dir = os.path.join(results_dir, 'scatter')
        util.check_filepath(scatter_dir, mkdir=True, verbose=verbose)

        plt.savefig(os.path.join(scatter_dir, 'MCMC_vs_SFR_DUST.pdf'), bbox_inches='tight')
        plt.close()
    if show:
        plt.show()


def incidence():
    return


def bin_profiles(galname, bin_method, grouping_dims = (6, 4), random_draw = False, show_binID = False, show_emline_masking = True, 
                 show = False, save = True, overwrite = True, verbose = False):
    plt.style.use(os.path.join(defaults.get_default_path('config'), 'figures.mplstyle'))

    datapath_dict = file_handler.init_datapaths(galname, bin_method, verbose = False)
    local_file = datapath_dict['LOCAL']
    cube_file = datapath_dict['LOGCUBE']
    
    with fits.open(cube_file) as cube:
        spatial_bins = cube['binid'].data[0]
        flux = cube['flux'].data 
        wave = cube['wave'].data 
        stellar_cont = cube['model'].data 
    
    with fits.open(local_file) as local_maps:
        redshift = local_maps['redshift'].data
        mcmc_cube = local_maps['mcmc_results'].data
        mcmc_16 = local_maps['mcmc_16th_perc'].data 
        mcmc_84 = local_maps['mcmc_84th_perc'].data 

        snrs = local_maps['nai_snr'].data
        ew = local_maps['ew_nai'].data
        ew_mask = local_maps['ew_nai_mask'].data

        ewnoem = local_maps['ew_noem'].data
        ewnoem_mask = local_maps['ew_noem_mask'].data

        vcen = local_maps['v_nai'].data
        vfrac = local_maps['v_nai_frac'].data
        vcen_mask = local_maps['v_nai_mask'].data

        vmax = local_maps['v_max_out'].data
        vmax_mask = local_maps['v_max_out_mask'].data

    delta_ew = ewnoem - ew
    datamask = (ew_mask + ewnoem_mask + vcen_mask + vmax_mask).astype(bool)

    vmaxlim = 100
    vcenlim = 50

    snr_llim = 45

    low_snr_llim = 20
    low_snr_ulim = 40

    delta_ew_llim = 0.1

    mask_arrays = {
        'outflow':(vmax > vmaxlim) & (snrs > snr_llim) & (~datamask),
        'inflow':(vcen > vcenlim) & (vfrac > 0.95) & (snrs > snr_llim) & (~datamask),
        'pcygni':(snrs >= 90) & (delta_ew > delta_ew_llim) & (delta_ew < 0.5) & (~datamask),
        'low_snrs':(snrs > low_snr_llim) & (snrs < low_snr_ulim) & (~datamask)
    }

    ## arrays to use for sorting if not using random select
    if not random_draw:
        sort_dict = {
            'inflow':ew.flatten(),
            'outflow':vmax.flatten(),
            'pcygni':snrs.flatten(),
            'low_snrs':None
        }

    title_dict = {
        'outflow':fr'$v_{{\mathrm{{max,\ out}}}} < -{vmaxlim}\ \mathrm{{km\ s^{{-1}}}}$',
        'inflow':fr'$v_{{\mathrm{{cen}}}} > {vcenlim}\ \mathrm{{km\ s^{{-1}}}}$',
        'pcygni':fr'$\mathrm{{PCygni}}\ \left( \mathrm{{\Delta EW > {delta_ew_llim}\ \AA}} \right)$',
        'low_snrs':fr'${low_snr_llim} < S/N_{{\mathrm{{Na\ D}}}} < {low_snr_ulim}$'
    }

    groupings = {}
    bin_array = []
    for key, value in mask_arrays.items():
        unique_bins, unique_inds = np.unique(spatial_bins[value], return_index=True)
        choice_size = int(np.prod(grouping_dims))
        if len(unique_bins) < choice_size:
            raise ValueError(f"Array '{key}' has {len(unique_bins)} values. Too small for {choice_size} plots.")
        
        if random_draw:
            select = np.random.choice(np.setdiff1d(unique_bins, bin_array), size=int(np.prod(grouping_dims)), replace=False)
        else:
            sort_array = sort_dict[key]
            if sort_array is not None:
                sort_inds = np.argsort(sort_array[unique_inds])
                sorted_bins = unique_bins[sort_inds]
            else:
                sorted_bins = unique_bins
            sorted_bins_select = np.setdiff1d(sorted_bins, bin_array)
            select = sorted_bins_select[:choice_size]
        groupings[key] = select

    COLORS = {
        "data": "#2c2a29",
        "model": "#0063ff",
        "bbox":'#b9fff4',
        "uncertainty": "#ce0014",
        "masked":'#ce0014',
        "vline": "#c92ad5",
        "hline": "#3327e7"
    }

    NaD_window = (5875, 5915)
    NaD_fit_window = (5880, 5910)
    NaD_rest = [5891.5833, 5897.5581]    

    base_w, base_h = plt.rcParams["figure.figsize"]

    n_plots = 4
    nrow, ncol = grouping_dims
    fig = plt.figure(figsize = (base_w * ncol * 1.5, base_h * nrow * 1.5))

    main_gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0.1)

    gs1 = main_gs[0,0].subgridspec(nrow, ncol, hspace = 0, wspace = 0)
    gs2 = main_gs[0,1].subgridspec(nrow, ncol, hspace = 0, wspace = 0)
    gs3 = main_gs[1,0].subgridspec(nrow, ncol, hspace = 0, wspace = 0)
    gs4 = main_gs[1,1].subgridspec(nrow, ncol, hspace = 0, wspace = 0)

    gs_plots = [gs1, gs2, gs3, gs4]

    axes_dict = {}

    for gs_idx, (gs, (key, bin_arr)) in enumerate(zip(gs_plots, groupings.items())):
        
        axes_dict[gs_idx] = {}

        top_ymax = 1.1 
        top_ymin = .9

        bottom_ymax = 1
        bottom_ymin = 0.9

        ax_group = fig.add_subplot(gs[:,:])
        ax_group.set_ylabel('Normalized Flux', fontsize=24)
        ax_group.set_xlabel(r'Rest Velocity $\mathrm{\left( km\ s^{-1} \right)}$', fontsize=24)
        ax_group.set_title(fr"{title_dict[key]}", fontsize=18)
        ax_group.tick_params(which = 'both', labelcolor='none', top=False, bottom=False, 
                            left=False, right=False)
        ax_group.spines['top'].set_visible(False)
        ax_group.spines['right'].set_visible(False)
        ax_group.spines['bottom'].set_visible(False)
        ax_group.spines['left'].set_visible(False)

        for row in range(nrow):
            axes_dict[gs_idx][row] = {}
            for col in range(ncol):
                inner = gridspec.GridSpecFromSubplotSpec(
                    2, 1,
                    subplot_spec=gs[row, col],
                    hspace=0,
                    height_ratios=[3.5,1]
                )
                ax_top = fig.add_subplot(inner[0])
                ax_bottom = fig.add_subplot(inner[1])

                axes_dict[gs_idx][row][col] = {
                    'top':ax_top,
                    'bottom':ax_bottom
                }

                binID = bin_arr[row + col]
                w = binID == spatial_bins
                ny, nx = np.where(w)
                y, x = ny[0], nx[0]

                ew = ewnoem[y,x]
                snr = snrs[y,x]
                vmaxout = vmax[y,x]

                lambda_0 = mcmc_cube[0, y, x]
                lambda_16 = mcmc_16[0, y, x]
                lambda_84 = mcmc_84[0, y, x]

                z = redshift[y, x]
                restwave = wave / (1 + z)
                flux_1D = flux[:, y, x]
                model_1D = stellar_cont[:, y, x]
                nflux = flux_1D / model_1D

                wave_window = (restwave >= NaD_window[0]) & (restwave <= NaD_window[1])
                inds = np.where(wave_window)

                # convert to velocity space
                lamblu0 = 5891.5833
                lamred0 = 5897.5581
                c = 2.998e5
                rest_velocity = ((restwave[inds] / lamred0) - 1) * c
                normflux = nflux[inds]

                v0 = ((lambda_0 / lamred0) - 1) * c
                v16 = ((lambda_16 / lamred0)) * c
                v84 = ((lambda_84 / lamred0)) * c

                vred0 = 0
                vblu0 = ((lamblu0 / lamred0) - 1) * c


                ax_top.plot(rest_velocity, normflux, color=COLORS['data'], drawstyle='steps-mid', linewidth = 1.3)

                textbox_kwargs = dict(va='top', color='#000000', fontsize=10, bbox = None)
                #bbox = dict(boxstyle='round',facecolor=COLORS['bbox'], edgecolor="#1f1e1e", alpha=.4)
                if show_binID:
                    ax_top.text(.95,.975, f"{int(binID)}", transform=ax_top.transAxes, ha='right', va='top', fontsize=7)
                    ax_top.text(0.05, 0.95, rf"$\mathrm{{EW}} = {ew:.2f}\ \mathrm{{\AA}}$"
                                "\n"
                                rf"$S/N = {snr:.0f}$", transform = ax_top.transAxes, ha='left', **textbox_kwargs)
                else:
                    ax_top.text(.05,.95, rf"$\mathrm{{EW}} = {ew:.2f}\ \mathrm{{\AA}}$", transform = ax_top.transAxes, ha='left', **textbox_kwargs)
                    ax_top.text(.95,.95, rf"$S/N = {snr:.0f}$", transform=ax_top.transAxes, ha='right', **textbox_kwargs)


                ax_top.fill_between([v0-v16, v0+v84], [-20, -20], [20, 20], color=COLORS['uncertainty'],
                            alpha=0.2)
                ax_top.vlines([v0], -20, 20, colors = '#000000', linestyles = 'dashed', linewidths = 1.)
                if vmaxout > 0 and v0 < 0:
                    ax_top.vlines([-vmaxout], -20, 20, colors=COLORS['model'], linestyles='dashed', linewidths = 1.0)

                ax_top.vlines([vblu0, vred0], -20, 20, colors = COLORS['vline'], linestyles = 'dotted', linewidths = 0.9)
                ax_bottom.vlines([vblu0, vred0], -20, 20, colors=COLORS['vline'], linestyles = 'dotted', linewidths = 0.9)

                ax_top.hlines([1.0], -2000 , 2000, colors=COLORS['hline'], linestyles = 'dotted', linewidths = 0.9)

                top_ymin = min(np.min(normflux) - .025, top_ymin)
                top_ymax = max(np.max(normflux) + .025, top_ymax)

                #ax_top.set_ylim(kwargs_dict[key]['ylim'])
                ax_top.set_xticklabels([])

                if show_emline_masking:
                    s=1
                    blim = [5850.0, 5870.0]
                    rlim = [5910.0, 5930.0]
                    bind = np.where((restwave > blim[0]) & (restwave < blim[1]))
                    rind = np.where((restwave > rlim[0]) & (restwave < rlim[1]))

                    continuum = np.concatenate((nflux[bind], nflux[rind]))

                    median = np.median(continuum)
                    standard_dev = np.std(continuum)

                    wave_window = (restwave >= NaD_fit_window[0]) & (restwave <= NaD_fit_window[1])
                    wave_in_window = restwave[wave_window]
                    vel_in_window = ((wave_in_window / lamred0) - 1) * c
                    flux_in_window = nflux[wave_window]

                    flux_masked = flux_in_window.copy()
                    mask = flux_in_window <= median + s * standard_dev # NOT MASKED
                    flux_masked[mask] = np.nan


                    ax_top.plot(vel_in_window, flux_masked, color=COLORS['masked'], drawstyle = 'steps-mid', lw=1.3)
                
                max_flux = max(flux_1D[inds].max(), model_1D[inds].max())
                min_flux = min(flux_1D[inds].min(), model_1D[inds].min())

                bottom_ymax = max(bottom_ymax, max_flux)

                obsflux_norm = flux_1D[inds] / max_flux#(flux_1D[inds] - flux_1D[inds].min()) / (flux_1D[inds].max() - flux_1D[inds].min())
                stellarmodel_norm = model_1D[inds] / max_flux#(model_1D[inds] - model_1D[inds].min()) / (model_1D[inds].max() - model_1D[inds].min())

                bottom_ymin = min(bottom_ymin, min(obsflux_norm.min(), stellarmodel_norm.min()))

                ax_bottom.plot(rest_velocity, obsflux_norm, color=COLORS['data'], drawstyle = 'steps-mid', lw=1.3)
                ax_bottom.plot(rest_velocity, stellarmodel_norm, color=COLORS['model'], drawstyle = 'steps-mid', lw=1.3)


                NaD_velocity_window = ( ((NaD_window[0] / lamred0) - 1) * c , ((NaD_window[1] / lamred0) - 1) * c )
                ax_top.set_xlim(NaD_velocity_window)
                ax_bottom.set_xlim(NaD_velocity_window)

                #ax_top.set_ylim(0.825, 1.1)
                #ax_bottom.set_ylim(0, 1.0)

                #ax_bottom.set_ylim(min(flux_1D[inds].min()-.01, model_1D[inds].min()-.01), max(flux_1D[inds].max()+.01, model_1D[inds].min()+.01))
                ax_bottom.set_yticklabels([])

                if row < nrow - 1:
                    ax_bottom.set_xticklabels([])
                if col > 0:
                    ax_top.set_yticklabels([])

        for row in range(nrow):
            for col in range(ncol):
                ax_top = axes_dict[gs_idx][row][col]['top']
                ax_bottom = axes_dict[gs_idx][row][col]['bottom']

                ax_top.set_ylim(np.floor(top_ymin * 40) / 40, np.ceil(top_ymax * 40) / 40) ## rounded to nearest 0.25

                ax_bottom.set_ylim(bottom_ymin, 1.0)
                    
    
    if save:
        output_dir = defaults.get_fig_paths(galname, bin_method, subdir = 'inspection')
        if overwrite:
            fname = f'{galname}-{bin_method}_bin_profiles.pdf'
        else:
            files = [f for f in os.listdir(output_dir) if f'bin_profiles' in f]
            fname = f'{galname}-{bin_method}_bin_profiles_{len(files)}.pdf'

        figpath = os.path.join(output_dir, fname)
        plt.savefig(figpath, bbox_inches='tight')
        util.verbose_print(verbose, f'Bin profiles figure saved to {figpath}')

    if show:
        plt.show()
    else:
        plt.close()
        



def plot_bin_profiles(galname, bin_method, bin_array = None, grouping = 'all', show = False, save = True, overwrite = True, show_emline_masking = True, verbose = False):
    if bin_array is not None: 
        raise ValueError("Custom bin input currently not supported")
        if bin_array.size > 60:
            raise ValueError(f"Shape of input bins not supported. Must be a 2D array with 4 columns and 1 - 6 rows.")
        
    plt.style.use(os.path.join(defaults.get_default_path('config'), 'figures.mplstyle'))

    datapath_dict = file_handler.init_datapaths(galname, bin_method, verbose = False)
    local_file = datapath_dict['LOCAL']
    cube_file = datapath_dict['LOGCUBE']
    
    with fits.open(cube_file) as cube:
        spatial_bins = cube['binid'].data[0]
        flux = cube['flux'].data 
        wave = cube['wave'].data 
        stellar_cont = cube['model'].data 
    
    with fits.open(local_file) as local_maps:
        redshift = local_maps['redshift'].data
        mcmc_cube = local_maps['mcmc_results'].data
        mcmc_16 = local_maps['mcmc_16th_perc'].data 
        mcmc_84 = local_maps['mcmc_84th_perc'].data 

        snrs = local_maps['nai_snr'].data
        ew = local_maps['ew_nai'].data
        ew_mask = local_maps['ew_nai_mask'].data

        ewnoem = local_maps['ew_noem'].data
        ewnoem_mask = local_maps['ew_noem_mask'].data

        vcen = local_maps['v_nai'].data
        vfrac = local_maps['v_nai_frac'].data
        vcen_mask = local_maps['v_nai_mask'].data

        vmax = local_maps['v_max_out'].data
        vmax_mask = local_maps['v_max_out_mask'].data

    delta_ew = ewnoem - ew
    datamask = (ew_mask + ewnoem_mask + vcen_mask + vmax_mask).astype(bool)
    
    groupings = {}
    kwargs_dict = {
        'inflow':dict(ylim=None),
        'outflow':dict(ylim=None),
        'pcygni':dict(ylim=(0.85,1.1)),
        'low_snrs':dict(ylim=None)
        }
    if bin_array is None:
        bin_array = []
        mask_arrays = {
            'inflow':(vmax > 0) & (snrs > 45) & (~datamask),
            'outflow':(vcen > 0) & (vfrac > 0.95) & (snrs > 45) & (~datamask),
            'pcygni':(snrs >= 90) & (delta_ew > 0.1) & (delta_ew < 0.5) & (~datamask),
            'low_snrs':(snrs > 20) & (snrs < 40) & (~datamask)
        }
        if grouping != 'all':
            if grouping not in list(mask_arrays.keys()):
                raise ValueError(f"grouping must be one of {list(mask_arrays.keys())}")
        for key, value in mask_arrays.items():
            if grouping != 'all' and key != grouping:
                continue
            unique_bins = np.unique(spatial_bins[value])
            random_select = np.random.choice(np.setdiff1d(unique_bins, bin_array), size=24, replace=False)
            groupings[key] = random_select
    else:
        groupings['custom'] = bin_array


    NaD_window = (5875, 5915)
    #NaD_window = (5880, 5920)
    NaD_rest = [5891.5833, 5897.5581]

    for key, bin_arr in groupings.items():
        util.verbose_print(verbose, f"Plotting line profiles for {len(bin_arr)} {key} bins")

        ncol = 4
        nrow = int(np.ceil(len(bin_arr) / ncol))

        base_w, base_h = plt.rcParams["figure.figsize"]
        fig = plt.figure(figsize=(ncol * base_w, nrow * base_h))

        outer = gridspec.GridSpec(nrow, ncol, figure=fig, hspace=0.05, wspace=0)

        ymin = .9
        ymax = 1.1
        # Loop only over rows
        axes = []
        for i in range(nrow):
            row_axes = []
            for j in range(ncol):
                # Each "cell" in outer grid gets its own 2-row subgrid
                inner = gridspec.GridSpecFromSubplotSpec(
                    2, 1,  # 2 vertical panels
                    subplot_spec=outer[i, j], 
                    hspace=-.125,
                    height_ratios=[3,1]
                )
                ax_top = fig.add_subplot(inner[0])
                ax_bottom = fig.add_subplot(inner[1])
                row_axes.append((ax_top, ax_bottom))
            axes.append(row_axes)

    
        for row in range(nrow):
            for col in range(ncol):
                binID = bin_arr[row + col]
                w = binID == spatial_bins
                ny, nx = np.where(w)
                y, x = ny[0], nx[0]

                ew = ewnoem[y,x]
                snr = snrs[y,x]

                lambda_0 = mcmc_cube[0, y, x]
                lambda_16 = mcmc_16[0, y, x]
                lambda_84 = mcmc_84[0, y, x]

                z = redshift[y, x]
                restwave = wave / (1 + z)
                flux_1D = flux[:, y, x]
                model_1D = stellar_cont[:, y, x]
                nflux = flux_1D / model_1D

                wave_window = (restwave >= NaD_window[0]) & (restwave <= NaD_window[1])
                inds = np.where(wave_window)

                # convert to velocity space
                lamblu0 = 5891.5833
                lamred0 = 5897.5581
                c = 2.998e5
                rest_velocity = ((restwave[inds] / lamred0) - 1) * c
                normflux = nflux[inds]

                ax_top = axes[row][col][0]
                ax_bottom = axes[row][col][1]

                v0 = ((lambda_0 / lamred0) - 1) * c
                v16 = ((lambda_16 / lamred0)) * c
                v84 = ((lambda_84 / lamred0)) * c

                vred0 = 0
                vblu0 = ((lamblu0 / lamred0) - 1) * c

                ['#75b8ce',
                '#3885d0',
                '#4a4fa5',
                '#302e4a',
                '#1f1e1e',
                '#4a252e',
                '#932e44',
                '#d34936',
                '#f18f51']

                ['#94f1f3',
                '#4bbae4',
                '#1e81dc',
                '#3c4e8e',
                '#22253b',
                '#000000',
                '#3a1d18',
                '#82322f',
                "#c74562",
                '#e487b4',
                '#fccdfc']
                
                ["#171717",
                '#0000cc',
                '#2600ff',
                '#7800ff',
                '#c92ad5',
                '#ff5ca3',
                '#ff906f',
                '#ffc23d',
                '#fff609']

                ['#b9fff4',
                '#5ac2ff',
                '#0063ff',
                '#3327e7',
                '#372d68',
                '#2c2a29',
                '#681922',
                '#ce0014',
                '#ff3e00',
                '#ff8201',
                '#ffcd83']
                COLORS = {
                        "data": "#2c2a29",
                        "model": "#0063ff",
                        "bbox":'#b9fff4',
                        "uncertainty": "#ce0014",
                        "masked":'#ce0014',
                        "vline": "#c92ad5",
                        "hline": "#3327e7"
                    }
                
                ax_top.plot(rest_velocity, normflux, color=COLORS['data'], drawstyle='steps-mid', linewidth = 1.3)
                ax_top.text(.95,.975, f"{int(binID)}", transform=ax_top.transAxes, ha='right', va='top', fontsize=7)
                bbox = dict(boxstyle='round',facecolor=COLORS['bbox'], edgecolor="#1f1e1e", alpha=.4)
                textbox_kwargs = dict(va='top', color='#000000', fontsize=10, bbox = None)
                #ax_top.text(.05,.1, rf"$\mathrm{{EW}} = {ew:.2f}\ \mathrm{{\AA}}$", transform = ax_top.transAxes, ha='left', **textbox_kwargs)
                #ax_top.text(.95,.1, rf"$S/N = {snr:.0f}$", transform=ax_top.transAxes, ha='right', **textbox_kwargs)
                ax_top.text(0.05, 0.95, rf"$\mathrm{{EW}} = {ew:.2f}\ \mathrm{{\AA}}$"
                            "\n"
                            rf"$S/N = {snr:.0f}$", transform = ax_top.transAxes, ha='left', **textbox_kwargs)

                ax_top.fill_between([v0-v16, v0+v84], [-20, -20], [20, 20], color=COLORS['uncertainty'],
                            alpha=0.2)
                ax_top.vlines([v0], -20, 20, colors = '#000000', linestyles = 'dashed', linewidths = 1.)

                ax_top.vlines([vblu0, vred0], -20, 20, colors = COLORS['vline'], linestyles = 'dotted', linewidths = 0.9)
                ax_bottom.vlines([vblu0, vred0], -20, 20, colors=COLORS['vline'], linestyles = 'dotted', linewidths = 0.9)

                ax_top.hlines([1.0], -2000 , 2000, colors=COLORS['hline'], linestyles = 'dotted', linewidths = 0.9)

                ymin = min(np.min(normflux) - .025, ymin)
                ymax = max(np.max(normflux) + .025, ymax)

                #ax_top.set_ylim(kwargs_dict[key]['ylim'])
                ax_top.set_xticklabels([])

                if show_emline_masking:
                    s=1
                    blim = [5850.0, 5870.0]
                    rlim = [5910.0, 5930.0]
                    bind = np.where((restwave > blim[0]) & (restwave < blim[1]))
                    rind = np.where((restwave > rlim[0]) & (restwave < rlim[1]))

                    continuum = np.concatenate((nflux[bind], nflux[rind]))

                    median = np.median(continuum)
                    standard_dev = np.std(continuum)

                    wave_window = (restwave >= NaD_window[0]) & (restwave <= NaD_window[1])
                    wave_in_window = restwave[wave_window]
                    vel_in_window = ((wave_in_window / lamred0) - 1) * c
                    flux_in_window = nflux[wave_window]

                    flux_masked = flux_in_window.copy()
                    mask = flux_in_window <= median + s * standard_dev # NOT MASKED
                    flux_masked[mask] = np.nan


                    ax_top.plot(vel_in_window, flux_masked, color=COLORS['masked'], drawstyle = 'steps-mid', lw=1.3)
                
                flux1_norm = (flux_1D[inds] - flux_1D[inds].min()) / (flux_1D[inds].max() - flux_1D[inds].min())
                flux2_norm = (model_1D[inds] - model_1D[inds].min()) / (model_1D[inds].max() - model_1D[inds].min())
                med_flux = round(max(np.median(flux_1D[inds]), np.median(model_1D[inds])), 2)
                ax_bottom.plot(rest_velocity, flux1_norm, color=COLORS['data'], drawstyle = 'steps-mid', lw=1.3)
                ax_bottom.plot(rest_velocity, flux2_norm, color=COLORS['model'], drawstyle = 'steps-mid', lw=1.3)


                #ax_bottom.set_ylim(min(flux_1D[inds].min()-.01, model_1D[inds].min()-.01), max(flux_1D[inds].max()+.01, model_1D[inds].min()+.01))

                if row < nrow - 1:
                    ax_bottom.set_xticklabels([])
                if col > 0:
                    ax_top.set_yticklabels([])
                    ax_bottom.set_yticklabels([])

        NaD_velocity_window = ( ((NaD_window[0] / lamred0) - 1) * c , ((NaD_window[1] / lamred0) - 1) * c )
        for row in range(nrow):
            for col in range(ncol):
                ax_top = axes[row][col][0]
                #ax_top.set_ylim(np.floor(ymin * 40) / 40, np.ceil(ymax * 40) / 40) ## rounded to nearest 0.25
                ax_top.set_ylim(0.825, 1.1)
                ax_top.set_xlim(NaD_velocity_window)
                ax_top.set_box_aspect(2/3)

                ax_bottom = axes[row][col][1]
                ax_bottom.set_xlim(NaD_velocity_window)
                ax_bottom.set_box_aspect(2/9)
                ax_bottom.set_ylim(-0.05,1.05)

        #print(ymin, ymax)
        fig.text(0.5, 0.08, r'Rest Velocity $\left( \mathrm{km\ s^{-1}} \right)$', ha='center',va='center', fontsize=25)
        fig.text(0.05, 0.5, r'Normalized Flux', ha='center', va='center', rotation='vertical', fontsize=25)


        if save:
            output_dir = defaults.get_fig_paths(galname, bin_method, subdir = 'inspection')
            if overwrite:
                fname = f'{galname}-{bin_method}_bin_inspect_{key}.pdf'
            else:
                files = [f for f in os.listdir(output_dir) if f'bin_inspect_{key}' in f]
                fname = f'{galname}-{bin_method}_bin_inspect_{len(files)}.pdf'

            figpath = os.path.join(output_dir, fname)
            plt.savefig(figpath, bbox_inches='tight')
            util.verbose_print(verbose, f'Bin inspection figure saved to {figpath}')
        if show:
            plt.show()
        else:
            plt.close()


def velocity_threshold_plots(galname, bin_method, threshold_data, vmap, vmap_error, vmap_mask = None, 
                    ewnoem = False, scatter_lim = 30, fig_save_dir = None, verbose=False):
    if fig_save_dir is not None:
        util.check_filepath(fig_save_dir, mkdir=False, verbose=verbose)
    plt.style.use(os.path.join(defaults.get_default_path(subdir='config'), 'figures.mplstyle'))

    outpath = defaults.get_fig_paths(galname, bin_method, 'inspection') if fig_save_dir is None else fig_save_dir
    util.check_filepath(outpath,mkdir=True,verbose=verbose)

    file_end = '_maskedem' if ewnoem else ''
    out_file = os.path.join(outpath, f'{galname}-{bin_method}_v_thresholds{file_end}.pdf')

    thresholds = file_handler.threshold_parser(galname, bin_method, require_ew=False)
    snranges = thresholds['sn_lims']
    ewlims = thresholds['ew_lims']

    datapath_dict = file_handler.init_datapaths(galname, bin_method, verbose=False, redshift=False)
    local_file = datapath_dict['LOCAL']
    with fits.open(local_file) as hdul:
        spatial_bins = hdul['spatial_bins'].data.flatten()
        unique_bins, bin_inds = np.unique(spatial_bins, return_index=True)

        snr = hdul['nai_snr'].data.flatten()[bin_inds]

        hduname = 'ew_noem' if ewnoem else 'ew_nai'
        ew = hdul[hduname].data.flatten()[bin_inds]
        ew_mask = hdul[f'{hduname}_mask'].data.flatten().astype(bool)[bin_inds]

    velocity = vmap.flatten()[bin_inds]
    #velocity_err = vmap_error.flatten()[bin_inds]
    if vmap_mask is not None:
        velocity_mask = vmap_mask.flatten()[bin_inds]
        velocity_mask[velocity_mask == 7] = 0
        velocity_mask = velocity_mask.astype(bool)
        datamask = np.logical_and(ew_mask, velocity_mask)
    else:
        datamask = ew_mask

    nrow = 2
    ncol = 2
    base_w, base_h = plt.rcParams["figure.figsize"]
    fig = plt.figure(figsize=(base_w * ncol, base_h * nrow))

    for i, (sn_low, sn_high) in enumerate(snranges):
        sn_low = int(sn_low) if np.isfinite(sn_low) else sn_low
        sn_high = int(sn_high) if np.isfinite(sn_high) else sn_high

        gs_parent = gridspec.GridSpec(nrow, ncol, figure=fig)
        subplot_spec = gs_parent[i]  # Get the SubplotSpec
        gs = subplot_spec.subgridspec(2, 1, height_ratios=[2,1], hspace=0)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        w = (snr > sn_low) & (snr <= sn_high)
        mask = np.logical_and(w, ~datamask)
        
        ew_bin = ew[mask]
        v_bin = velocity[mask]
        
        ax1.scatter(ew_bin, v_bin, c='dimgray', s=1, alpha=0.9)
        temp_df = pd.DataFrame({
            'EW':np.asarray(ew_bin, dtype='<f8'),
            'vel':np.asarray(v_bin, dtype='<f8')
             })
        
        sns.histplot(data=temp_df, x = 'EW', y = 'vel', bins=60, ax=ax1, pthresh=0.1, cmap='mako')
        if np.isfinite(ewlims[i]):
            ax1.vlines(ewlims[i], -1000, 1000, colors='#c92ad5', linestyles='dotted', linewidths=1.5, label=rf'{ewlims[i]:.1f} $\mathrm{{\AA}}$')
            ax2.vlines(ewlims[i], -1000, 1000, colors='#c92ad5', linestyles='dotted', linewidths=1.5)
            ax1.legend(fancybox=True, loc='upper right', fontsize=12)
        ax1.grid(visible=True, linewidth=0.5, zorder=0, alpha = 0.25)
        ax2.grid(visible=True, linewidth=0.5, zorder=0, alpha = 0.25)

        ax1.set_ylim(-650, 650)
        if ewnoem:
            ax1.set_xlim(0,3)
        else:
            ax1.set_xlim(-1, 3)
        ax1.set_xticklabels([])
        #ax1.tick_params(labelsize=12)

        title = rf"${sn_low} < S/N \leq {sn_high}$" if np.isfinite(sn_high) else rf"$S/N > {sn_low}$"
        ax1.set_title(title, fontsize=14)
        #ax1.set_ylabel(r'$v_{\mathrm{cen}}\ \mathrm{(km\ s^{-1})}$',fontsize=14)
        ax1.set_ylabel('')

        
        key = f"{sn_low}-{sn_high}"
        subdict = threshold_data[key]
        v_std = subdict['std']
        med_ew = subdict['medew']
        ax2.plot(med_ew, v_std, drawstyle='steps-mid', color='dimgray')
        ax2.set_yscale('log')
        ax2.hlines([scatter_lim], -10, 10, colors='k', linestyles='dashed', linewidths = 1)

        ax2.set_ylim(0, 1000)
        if ewnoem:
            ax2.set_xlim(0,3)
        else:
            ax2.set_xlim(-1, 3)
        #ax2.tick_params(labelsize=12)

        #ax2.set_ylabel(r'$\mathrm{med}\ \sigma_{v_{\mathrm{cen}}}$', fontsize=14)
        if i == 0 or i == 2:
            fig.text(-.3, 0.5, r'$v_{\mathrm{cen}}\ \mathrm{(km\ s^{-1})}$', transform=ax1.transAxes, rotation = 'vertical', ha='center', va='center', fontsize=14)
            fig.text(-.3, 0.5, r'$\mathrm{med}\ \sigma_{v_{\mathrm{cen}}}$', transform=ax2.transAxes, rotation = 'vertical', ha='center', va='center', fontsize=14)
        if i == 1 or i == 3:
            ax1.set_ylabel('')
            ax2.set_ylabel('')
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])
        if i == 0 or i == 1:
            ax2.set_xticklabels([])


    fig.text(0.5, 0.05, r'$\mathrm{EW_{Na\ D}}\ (\mathrm{\AA})$', ha='center',va='center', fontsize=16)
    plt.savefig(out_file, bbox_inches='tight')
    util.verbose_print(verbose, f"Saving velocity scatter figure to {out_file}")


def plot_BPT_MAP(galname, bin_method, show = False, save = True, verbose=False):
    datpath_dict = file_handler.init_datapaths(galname, bin_method)
    with fits.open(datpath_dict['LOCAL']) as hdu:
        bpt_map = hdu['BPT'].data.astype(float)

    classification_int = {
        'Star-forming': 1,
        'Composite': 2,
        'Seyfert': 3,
        'LINER': 4,
        'Ambiguous': 5
        }
    
    colors = ['#3498db',  # Star-forming - bright blue
          '#16a085',  # Composite - teal/cyan
          '#e74c3c',  # Seyfert - red
          '#f39c12',  # LINER - orange
          '#95a5a6']  # Ambiguous - gray
    
    cmap = ListedColormap(colors)
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

    bpt_map[bpt_map==0] = np.nan
    im = ax.imshow(bpt_map, cmap=cmap, norm=norm, origin='lower', 
                interpolation='nearest', aspect='auto', extent=[32.4, -32.6, -32.4, 32.6])

    legend_elements = [
        Patch(facecolor=colors[0], label='Star-forming'),
        Patch(facecolor=colors[1], label='Composite'),
        Patch(facecolor=colors[2], label='Seyfert'),
        Patch(facecolor=colors[3], label='LINER'),
        Patch(facecolor=colors[4], label='Ambiguous')
    ]

    ax.legend(handles=legend_elements, loc='upper right', 
            frameon=True, fancybox=False, shadow=False, title='BPT Classification')

    ax.set_xlabel(r'$\Delta \alpha$ (arcsec)')
    ax.set_ylabel(r'$\Delta \delta$ (arcsec)')
    ax.set_facecolor('lightgray')
    plt.tight_layout()

    if save:
        resultsdir = defaults.get_fig_paths(galname, bin_method, subdir='results')
        savedir = os.path.join(resultsdir, 'maps')
        plt.savefig(os.path.join(savedir, f"{galname}-{bin_method}_BPT_MAP.pdf"))
    if show:
        plt.show()
    else:
        plt.close()

    # Optional: Print statistics
    # print("BPT Classification Statistics:")
    # for name, value in classification_int.items():
    #     count = np.sum(bpt_map == value)
    #     percentage = (count / bpt_map.size) * 100
    #     print(f"{name:15s}: {count:5d} pixels ({percentage:5.2f}%)")


def incidence():
    return 

# def corner_plotter(galname, bin_method, bin_list, show = False, save = True, verbose = False):
#     datapath_dict = file_handler.init_datapaths(galname, bin_method)
#     mcmc_files = datapath_dict['MCMC']

#     with fits.open(datapath_dict['LOGCUBE']) as logcube:
#         flux_cube = logcube['FLUX'].data
#         model_cube = logcube['MODEL'].data
#         wavelength = logcube['WAVE'].data
    
#     with fits.open(datapath_dict['LOCAL']) as local:
#         redshift = local['REDSHIFT'].data
#         binmap = local['SPATIAL_BINS'].data

#     sorted_paths = sort_paths(mcmc_files)

#     NaD_window = (5875, 5915)

#     for binID in bin_list:
#         util.verbose_print(verbose, f"Obtaining samples for bin {binID}")
#         w = binID == binmap
#         ny,nx = np.where(w)
#         y,x = ny[0], nx[0]

#         z = redshift[y,x]
#         flux_bin = flux_cube[:,y,x]
#         stellar_flux_bin = model_cube[:,y,x]
#         restwave_bin = wavelength/(1+z)

#         NaD_lims = (restwave_bin>=NaD_window[0]) & (restwave_bin<=NaD_window[1])

#         flux = flux_bin[NaD_lims]
#         stellar_flux = stellar_flux_bin[NaD_lims]
#         restwave = restwave_bin[NaD_lims]

#         normflux = flux/stellar_flux
#         for mcmc_fil in sorted_paths:
#             path, file = os.path.split(mcmc_fil)
#             match = re.search(r'binid-(\d+)-(\d+)-samples', file)

#             if match:
#                 start_ID = int(match.group(1))
#                 end_ID = int(match.group(2))

#                 if start_ID <= binID <= end_ID:
#                     util.verbose_print(verbose, f"    File found {file}")
#                     data = fits.open(mcmc_fil)
#                     data_table = Table(data[1].data)

#                     all_bins = data_table['bin'].data

#                     i = np.where(binID == all_bins)[0][0]
#                     samples = data_table[i]['samples']
#                     percentiles = data_table[i]['percentiles']
#                     theta = percentiles[:,0]
#                     model_dict = model_nai.model_NaI(theta, z, restwave)

#                     flat_samples = samples[:,1000:,:].reshape(-1, 4)
#                     labels = [r'$\lambda$', r'$\mathrm{log}N$', r'$b_D$', r'$C_f$']

#                     corner_fig = corner.corner(
#                             flat_samples,
#                             labels=labels,
#                             show_titles=True,
#                             title_fmt=".2f",
#                             quantiles=[0.16, 0.5, 0.84]
#                             #title_kwargs={"fontsize": 12}
#                     )

#                     # Single flux + model plot on the right  
#                     ax_spec = corner_fig.add_axes([0.7, 0.7, 0.25, 0.25]) # [left, bottom, width, height] in figure coords

#                     # Plot observed flux and best-fit model on same axes
#                     ax_spec.plot(restwave, normflux, 'k', linewidth=1, drawstyle='steps-mid')
#                     ax_spec.plot(model_dict['modwv'], model_dict['modflx'], 'b', 
#                                 linewidth=1.5, label='Best Fit Model')

#                     ax_spec.set_xlabel(r'Wavelength $\left( \mathrm{\AA} \right)$')
#                     ax_spec.set_ylabel('Normalized Flux')
#                     #ax_spec.legend()
#                     ax_spec.grid(True, alpha=0.3)
#                     ax_spec.set_xlim(NaD_window)

#                     # Optional: add residuals as small subplot below
#                     # ax_resid = fig.add_subplot(gs[1], ...)
#                     # ax_resid.plot(wavelength, observed_flux - model_flux, 'gray')

#                     corner_fig.suptitle(f"Bin {binID}")
#                     plt.show()

#                     if save:
#                         output_dir = defaults.get_fig_paths(galname, bin_method, subdir = 'inspection')
#                         fname = f"Samples_corner_bin_{binID}.pdf"
#                         figpath = os.path.join(output_dir, fname)
#                         plt.savefig(figpath, bbox_inches='tight')
#                         util.verbose_print(verbose, f"   Figure saved to {figpath}\n")
#                         plt.close()
#                     break
#         else:
#             print(f'No file found for bin {binID}\n')

# def chain_plotter(galname, bin_method, bin_list, show = False, save = True, verbose = False):
#     from chainconsumer import ChainConsumer, Chain, PlotConfig, Truth
#     import pandas as pd

#     datapath_dict = file_handler.init_datapaths(galname, bin_method)
#     mcmc_files = datapath_dict['MCMC']

#     with fits.open(datapath_dict['LOGCUBE']) as logcube:
#         flux_cube = logcube['FLUX'].data
#         model_cube = logcube['MODEL'].data
#         wavelength = logcube['WAVE'].data
    
#     with fits.open(datapath_dict['LOCAL']) as local:
#         redshift = local['REDSHIFT'].data
#         binmap = local['SPATIAL_BINS'].data

#     sorted_paths = sort_paths(mcmc_files)

#     NaD_window = (5875, 5915)

#     for binID in bin_list:
#         util.verbose_print(verbose, f"Obtaining samples for bin {binID}")
#         w = binID == binmap
#         ny,nx = np.where(w)
#         y,x = ny[0], nx[0]

#         z = redshift[y,x]
#         flux_bin = flux_cube[:,y,x]
#         stellar_flux_bin = model_cube[:,y,x]
#         restwave_bin = wavelength/(1+z)

#         NaD_lims = (restwave_bin>=NaD_window[0]) & (restwave_bin<=NaD_window[1])

#         flux = flux_bin[NaD_lims]
#         stellar_flux = stellar_flux_bin[NaD_lims]
#         restwave = restwave_bin[NaD_lims]

#         normflux = flux/stellar_flux
#         for mcmc_fil in sorted_paths:
#             path, file = os.path.split(mcmc_fil)
#             match = re.search(r'binid-(\d+)-(\d+)-samples', file)

#             if match:
#                 start_ID = int(match.group(1))
#                 end_ID = int(match.group(2))

#                 if start_ID <= binID <= end_ID:
#                     util.verbose_print(verbose, f"    File found {file}")
#                     data = fits.open(mcmc_fil)
#                     data_table = Table(data[1].data)

#                     all_bins = data_table['bin'].data

#                     i = np.where(binID == all_bins)[0][0]
#                     samples = data_table[i]['samples']
#                     percentiles = data_table[i]['percentiles']
#                     theta = percentiles[:,0]
#                     model_dict = model_nai.model_NaI(theta, z, restwave)

#                     flat_samples = samples[:,1000:,:].reshape(-1, 4)
#                     labels_units = [r'$\lambda_{0}\ \left( \mathrm{\AA} \right)$', r'$\mathrm{log}N\ \left( \mathrm{cm^{-2}} \right)$', r'$b_D\ \left( \mathrm{km\ s^{-1}} \right)$', r'$C_f$']
#                     labels = [r'$\lambda_{0}$', r'$\mathrm{log}N$', r'$b_D$', r'$C_f$']
#                     param_names = ['lambda', 'logN', 'bD', 'Cf']

#                     df_samples = pd.DataFrame(flat_samples, columns = param_names)

#                     c = ChainConsumer()
#                     chain = Chain(
#                         samples = df_samples,
#                         name = f"Bin {binID}",
#                         color = 'gray',
#                         shade_gradient=2,
#                         sigmas = np.linspace(0,1,8).tolist(),
#                         smooth=2,
#                         bins=25,
#                         plot_point = True,
#                         plot_cloud = True,
#                         marker_style = 'x',
#                         marker_size = 100,
#                         num_cloud = 10000,
#                         shade = True,
#                         linewidth = 1,
#                         show_contour_labels = False,
#                         linestyle='-',
#                         drawstyles = 'steps'
#                     )
#                     c.add_chain(chain)
#                     truth_dict = dict(zip(param_names, theta))
#                     c.add_truth(Truth(location=truth_dict))
#                     param_label_map = dict(zip(param_names, labels_units))
#                     c.set_plot_config(
#                         PlotConfig(
#                             labels=param_label_map,
#                             #plot_hists=True,
#                             extents={'lambda':(5896.6,5897.1), 'logN':(13, 14.), 'bD':(60,110), 'Cf':(0.25, 0.6)},
#                             label_font_size=16
#                             )
#                     )
#                     fig = c.plotter.plot()

#                     # Single flux + model plot on the right  
#                     ax_spec = fig.add_axes([0.7, 0.7, 0.225, 0.225]) # [left, bottom, width, height] in figure coords

#                     # Plot observed flux and best-fit model on same axes
#                     ax_spec.plot(restwave, normflux, 'k', linewidth=1, drawstyle='steps-mid')
#                     ax_spec.plot(model_dict['modwv'], model_dict['modflx'], 'b', 
#                                 linewidth=1.5, label='Best Fit Model')

#                     ax_spec.set_xlabel(r'Wavelength $\left( \mathrm{\AA} \right)$')
#                     ax_spec.set_ylabel('Normalized Flux')
#                     ax_spec.grid(True, alpha=0.3)
#                     ax_spec.set_xlim(NaD_window)
#                     if show:
#                         plt.show()

#                     if save:
#                         output_dir = defaults.get_fig_paths(galname, bin_method, subdir = 'inspection')
#                         fname = f"Samples_corner_bin_{binID}.pdf"
#                         figpath = os.path.join(output_dir, fname)
#                         plt.savefig(figpath, bbox_inches='tight')
#                         util.verbose_print(verbose, f"   Figure saved to {figpath}\n")
#                         plt.close()
#                     break
#         else:
#             print(f'No file found for bin {binID}\n')