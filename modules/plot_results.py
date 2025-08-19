import os
import numpy as np
from astropy.io import fits

import argparse

from scipy.optimize import curve_fit
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
#import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
#from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from modules import util, defaults, file_handler, plotter, inspect
from modules.util import verbose_print

import plotly.graph_objects as go
#from plotly.subplots import make_subplots
import plotly.io as pio

import string

def plot_dap_maps(galname: str, bin_method: str, output_dir = None, verbose = False):
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


    maps = fits.open(mapsfile)

    pipepline_datapath = defaults.get_data_path('pipeline')
    muse_cubes_path = os.path.join(pipepline_datapath, 'muse_cubes')
    raw_cube_path = os.path.join(muse_cubes_path, galname, f"{galname}.fits")

    rgb = rgb_im(raw_cube_path)
    snr = maps['bin_snr'].data
    radius = maps['bin_lwellcoo'].data[1]

    chisq = maps['stellar_fom'].data[2]
    stellar_vel = maps['stellar_vel'].data
    stellar_sigma = maps['stellar_sigma'].data

    d4000 = maps['specindex'].data[43]

    emlines = maps['EMLINE_GFLUX'].data
    ha = emlines[23]
    hb = emlines[14]
    oiii = emlines[16] #oiii 5007 
    #sii = emlines[25] + emlines[26] # sii 6718,32
    #oi = emlines[20] # oi 6302
    #nii = emlines[24] # nii 6585

    plotdicts = {
        'RGB':dict(image = rgb, cmap = None, vmin = None, vmax = None, v_str = 'B, V, R'),
        'RADIUS':dict(image = radius, cmap = 'autumn_r', vmin=0, vmax=1, v_str = r'$R / R_e$'),
        'SNR':dict(image = snr, cmap = 'coolwarm', vmin=0, vmax=75, v_str = r'$S/N_g$'),

        'STELLAR_VEL':dict(image = stellar_vel, cmap = 'seismic', vmin = -250, vmax = 250, v_str = r'$V_{\star}\ \left( \mathrm{km\ s^{-1}} \right)$'),
        'STELLAR_SIG':dict(image = stellar_sigma, cmap = 'inferno', vmin = 0, vmax = 150, v_str = r'$\sigma_{\star}\ \left( \mathrm{km\ s^{-1}} \right)$'),
        'CHISQ':dict(image = chisq, cmap = 'cividis', vmin = 0, vmax = 2, v_str = r'$\chi^2_{\nu}$'),
        #'D4000':dict(image = d4000, cmap = 'berlin', vmin = 1.0, vmax = 2.1, v_str = r'$\mathrm{D4000}$'),

        'H_alpha':dict(image = ha, cmap = 'viridis', vmin = 0, vmax = 10, v_str = r'$\mathrm{H\alpha}\ \left( \mathrm{10^{-17}\ erg\ s^{-1}\ cm^{-2}\ spaxel^{-1}} \right)$'),
        'H_beta':dict(image = hb, cmap = 'viridis', vmin = 0, vmax = 2, v_str = r'$\mathrm{H\beta}\ \left( \mathrm{10^{-17}\ erg\ s^{-1}\ cm^{-2}\ spaxel^{-1}} \right)$'),
        'OIII':dict(image = oiii, cmap = 'viridis', vmin = 0, vmax = 1, v_str = r'$[\mathrm{O\,III}]\ \left( \mathrm{10^{-17}\ erg\ s^{-1}\ cm^{-2}\ spaxel^{-1}} \right)$')
        #'OI':dict(image = oi, cmap = 'viridis', vmin = 0, vmax = 2, v_str = r'$\mathrm{OI}\ \left( \mathrm{10^{-17}\ erg\ s^{-1}\ cm^{-2}\ spaxel^{-1}} \right)$'),
        #'SII':dict(image = sii, cmap = 'viridis', vmin = 0, vmax = 2, v_str = r'$\mathrm{SII}\ \left( \mathrm{10^{-17}\ erg\ s^{-1}\ cm^{-2}\ spaxel^{-1}} \right)$'),
        #'NII':dict(image = nii, cmap = 'viridis', vmin = 0, vmax = 2, v_str = r'$\mathrm{NII}\ \left( \mathrm{10^{-17}\ erg\ s^{-1}\ cm^{-2}\ spaxel^{-1}} \right)$')
    }
    
    alphabet = list(string.ascii_lowercase)

    #fig, ax = plt.subplots(3,3,sharex=True, sharey=True, figsize=(12,12))

    nrow = 3
    ncol = 3

    # Setup figure and GridSpec
    fig = plt.figure(figsize=(ncol*4, nrow*4))
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
        if key != "RGB":
            plotmap[plotmap == 0] = np.nan
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
            cbar.set_label(value_string, fontsize=14, labelpad=-25)
            cbar.outline.set_visible(False)
            cbar.ax.patch.set_alpha(0)
        else:
            def smart_int_formatter(x, pos):
                if abs(x - round(x)) < 1e-2:  # very close to integer
                    return f'{int(round(x))}'
                else:
                    return ''  # empty string means no label
                
            cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
            cbar.set_label(value_string, fontsize=12, labelpad=-45)
            cax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_major_formatter(FuncFormatter(smart_int_formatter))


        a.text(0.075, 0.9, f'({char})', fontsize=10, transform=a.transAxes, color='white')

    for i, ax in enumerate(axes):
        row, col = divmod(i, nrow)
        if row < nrow-1:  # Hide x ticks except for bottom row
            ax.set_xticklabels([])
        if col > 0:  # Hide y ticks except for left column
            ax.set_yticklabels([])

    # Axis labels for the whole figure
    # fig.text(0.5, 0.05, r'$\Delta \alpha$ (arcsec)', ha='center', va='center', fontsize=20)
    # fig.text(0.11, 0.5, r'$\Delta \delta$ (arcsec)', ha='center', va='center', rotation='vertical', fontsize=20)
    fig.supxlabel(r'$\Delta \alpha$ (arcsec)', fontsize=20)
    fig.supylabel(r'$\Delta \delta$ (arcsec)', fontsize=20)    

    if output_dir is None:
        output_dir = defaults.get_fig_paths(galname, bin_method, subdir='dap')

    # Save output
    outfile = os.path.join(output_dir, f"{galname}-{bin_method}_dapmaps.pdf")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(outfile, bbox_inches='tight')


def plot_local_maps(galname, bin_method, local_maps_path, output_dir = None, verbose=False):
    hdul = fits.open(local_maps_path)
    if output_dir is None:
        output_dir = defaults.get_fig_paths(galname, bin_method, subdir='results')

    util.verbose_print(verbose, "Creating individual map plots...")

    plotter.map_plotter(galname, bin_method, hdul['nai_snr'].data, 'NAI_SNR', output_dir, r'$S/N_{\mathrm{Na\ D}}$', '', 'managua',
                0, 100, histogram=True, verbose=verbose)
    
    plotter.map_plotter(galname, bin_method, hdul['ew_nai'].data, 'EW_NAI', output_dir, r'$\mathrm{EW_{Na\ D}}$', r'$\left( \mathrm{\AA} \right)$', 'rainbow',
                -0.5, 2, mask = hdul['ew_nai_mask'].data, histogram=True, verbose=verbose)
    
    plotter.map_plotter(galname, bin_method, hdul['ew_noem'].data, 'EW_NOEM', output_dir, r'$\mathrm{EW_{Na\ D}}$', r'$\left( \mathrm{\AA} \right)$', 'rainbow',
                -0.5, 2, mask = hdul['ew_noem_mask'].data, histogram=True, verbose=verbose)
    
    plotter.map_plotter(galname, bin_method, hdul['sfrsd'].data, 'SFRSD', output_dir, r"$\mathrm{log \Sigma_{SFR}}$", r"$\left( \mathrm{M_{\odot}\ kpc^{-2}\ yr^{-1}\ spaxel^{-1}} \right)$",
                'rainbow', -2.5, 0, mask = hdul['sfrsd_mask'].data, histogram=True, verbose=verbose)
    
    plotter.map_plotter(galname, bin_method, hdul['v_nai'].data, 'V_NaI_masked', output_dir, r"$v_{\mathrm{Na\ D}}$", r"$\left( \mathrm{km\ s^{-1}} \right)$", 'seismic', 
                -250, 250, mask=hdul['v_nai_mask'].data, minmax=True, mask_ignore_list=[8], histogram=True, verbose=verbose)
    
    plotter.map_plotter(galname, bin_method, hdul['v_nai'].data, 'V_NaI', output_dir, r"$v_{\mathrm{cen}}$", r"$\left( \mathrm{km\ s^{-1}} \right)$", 'seismic', 
                -200, 200, minmax=True, histogram=True, verbose=verbose)
    


def plot_local_grid(galname: str, bin_method: str, local_maps_path: str, mask = True, output_dir = None, verbose = False):
    util.verbose_print(verbose, "Creating grid plot...")
    hdul = fits.open(local_maps_path)
    
    
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

    #z = hdul['redshift'].data
    #v_frac = hdul['V_NaI_FRAC'].data
    # mcmc = hdul['mcmc_results'].data
    # logn = mcmc[1]
    # bd = mcmc[2]
    # cf = mcmc[3]

    plotdicts = {
        'SNR':dict(image = snr, mask = snr_mask, cmap = 'managua', vmin = 0, vmax = 100, v_str = r'$S/N_{\mathrm{Na\ D}}$'),
        'EW':dict(image = ew, mask = ew_mask, cmap = 'rainbow', vmin=-0.2, vmax=2, v_str = r'$\mathrm{EW_{Na\ D}}\ \left( \mathrm{\AA} \right)$'),

        'SFRSD':dict(image = sfrsd, mask = sfrsd_mask, cmap = 'rainbow', vmin=-2.5, vmax=0, v_str = r'$\mathrm{log\ \Sigma_{SFR}}\ \left( \mathrm{M_{\odot}\ kpc^{-2}\ yr^{-1}\ spaxel^{-1}} \right)$'),
        'V_BULK':dict(image = v_bulk, mask = v_mask, cmap = 'seismic', vmin = -250, vmax = 250, v_str = r'$v_{\mathrm{cen}}\ \left( \mathrm{km\ s^{-1}} \right)$'),
    }
    
    alphabet = list(string.ascii_lowercase)

    nrow = 2
    ncol = 2

    # Setup figure and GridSpec
    fig = plt.figure(figsize=(ncol*4, nrow*4))
    gs = GridSpec(nrow, ncol, figure=fig, hspace=.375, wspace=0)

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

        plotmap[plotmask] = np.nan
        # TODO mask???
        im = a.imshow(plotmap, origin='lower',
                    vmin=plot_dict['vmin'], vmax=plot_dict['vmax'],
                    cmap=plot_dict['cmap'], extent=[32.4, -32.6, -32.4, 32.6])

        a.set_facecolor('lightgray')

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
        cbar.set_label(value_string, fontsize=12, labelpad=-45)
        cax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(smart_int_formatter))


        a.text(0.075, 0.9, f'({char})', fontsize=10, transform=a.transAxes, color='white')

    for i, ax in enumerate(axes):
        row, col = divmod(i, nrow)
        if row < nrow-1:  # Hide x ticks except for bottom row
            ax.set_xticklabels([])
        if col > 0:  # Hide y ticks except for left column
            ax.set_yticklabels([])

    # Axis labels for the whole figure
    #fig.text(0.5, 0.05, r'$\Delta \alpha$ (arcsec)', ha='center', va='center', fontsize=20)
    #fig.text(0.07, 0.5, r'$\Delta \delta$ (arcsec)', ha='center', va='center', rotation='vertical', fontsize=20)
    fig.supxlabel(r'$\Delta \alpha$ (arcsec)', fontsize=20)
    fig.supylabel(r'$\Delta \delta$ (arcsec)', fontsize=20)

    # Save output
    if output_dir is None:
        output_dir = defaults.get_fig_paths(galname, bin_method, subdir='results')
    outfile = os.path.join(output_dir, f"{galname}-{bin_method}_results_mapgrid.pdf")
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(outfile, bbox_inches='tight')
    util.verbose_print(verbose, f"Results map grid saved to: {outfile}")


def velocity_vs_sfr(galname, bin_method, local_maps_path, output_dir = None, pearson = True, contours = True,
                    radius_cbar = False, hists = False, verbose = False):

    hdul = fits.open(local_maps_path)

    spatial_bins = hdul['SPATIAL_BINS'].data
    radius_map = hdul['RADIUS'].data

    vmap = hdul['V_NaI'].data
    vmap_mask = hdul['V_NaI_MASK'].data
    vmap_error = np.mean(hdul['V_NaI_ERROR'].data, axis=0)

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

    ## colormap and errorbar style
    if radius_cbar:
        normalized = mcolors.Normalize(vmin = np.min(radii), vmax = np.max(radii))
        cmap = cm.Oranges
        colors = cmap(normalized(radii))
    else:
        normalized = mcolors.Normalize(vmin = np.min(velocities), vmax = np.max(velocities))
        cmap = cm.bwr
        colors = cmap(normalized(velocities))

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
    for sf, vel, c in zip(sfrs, velocities, colors):
        scatter = ax.scatter(sf, vel, facecolors=c, **scatterstyle)
    
    x_pos = 0.1
    y_pos = 0.1
    x_pos_data, y_pos_data = ax.transData.inverted().transform(ax.transAxes.transform((x_pos, y_pos)))
    ax.errorbar(x_pos_data, y_pos_data, yerr=np.mean(velocity_errors), fmt='none', color='black', capsize=3, elinewidth=1.5)


    sm = cm.ScalarMappable(cmap=cmap, norm=normalized)
    sm.set_array([])  # Dummy array for colorbar
    #cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    #cbar.set_label(label=r"$R / R_e$", rotation=270, labelpad=21)

    ax.set_xlim(-2.6,-.1)
    ax.set_xlabel(r'$\mathrm{log\ \Sigma_{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spaxel^{-1}} \right)$')
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
        output_dir = defaults.get_fig_paths(galname, bin_method, subdir='results')

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
            xaxis_title=r"$ \mathrm{log\ \Sigma_{SFR}} \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}} \right)$",
            yaxis_title=r"$ v_{\mathrm{Na\ D}}\ \left( \mathrm{km\ s^{-1}} \right) $",
            width=600,
            height=600,
            margin=dict(t=15, b=15, l=15, r=15)
        )

        fig.update_xaxes(
            range=[-2.5, 0],
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
            range=[-250, 250],
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


def terminal_velocity(galname, bin_method, local_maps_path, output_dir = None, radius_cbar = False, power_law = True, verbose = True):
    # open local maps fits file
    hdul = fits.open(local_maps_path)

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

    items = zip(terminal_velocities, sfrs, colors)
    #iterator = tqdm(items, desc="Drawing Terminal Velocity vs SFR figure") if verbose else items
    for tv, sfr, c in items:
        #plt.errorbar(sfr, tv, xerr=None, yerr=tv_err, color = c, **style)
        sc = ax.scatter(sfr, tv, marker='o', s = 12, color=c, ec='k', lw=.75)

    x_pos = 0.01
    y_pos = 0.1
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

        modsfr = np.logspace(-3, 0, 1000)
        modv = wind_model(modsfr, popt[0], popt[1])

        #model_label = rf'$v = {popt[0]:.0f}\ \left( \Sigma_{{\mathrm{{SFR}}}} \right)^{{{popt[1]:.2f}}}$'
        model_label = rf'$v \propto \Sigma_{{\mathrm{{SFR}}}} ^{{{popt[1]:.2f} \pm {np.sqrt(pcov[1,1]):.2f}}}$'
        #model_label = rf'$\alpha = {popt[1]:.2f} \pm {np.sqrt(pcov[1,1]):.2f}$'
        ax.plot(np.log10(modsfr), modv, 'dimgray', linestyle='dashed', 
                label=model_label)

        ax.legend(frameon=False, fontsize=17)

    ax.set_xlim(-3,0)
    ax.set_xlabel(r'$\mathrm{log}\ \Sigma_{\mathrm{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}} \right)$')
    ax.set_ylabel(r'$ v_{\mathrm{out,\ max}}\ \left( \mathrm{km\ s^{-1}} \right)$')

    if output_dir is None:
        output_dir = defaults.get_fig_paths(galname, bin_method, subdir='results')

    outfil = os.path.join(output_dir, f'{galname}-{bin_method}-terminalv_vs_sfr.pdf')
    plt.savefig(outfil, bbox_inches='tight')
    verbose_print(verbose, f"Terminal velocity vs SFR fig saved to {outfil}")    



def get_args():
    parser = argparse.ArgumentParser(description="A script to create/overwrite plots without rerunning analyze_NaI")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)

    return parser.parse_args()


def main(args):
    galname = args.galname
    bin_method = args.bin_method
    verbose = args.verbose
    analysis_plans = defaults.analysis_plans()
    corr_key = 'BETA-CORR'

    local_data = defaults.get_data_path(subdir='local')
    local_outputs = os.path.join(local_data, 'local_outputs', f"{galname}-{bin_method}", corr_key, analysis_plans)
    local_maps_path = os.path.join(local_outputs, f"{galname}-{bin_method}-local_maps.fits")
    mplconfig = os.path.join(defaults.get_default_path('config'), 'figures.mplstyle')
    plt.style.use(mplconfig)

    verbose_print(verbose, f"Creating Plots for {galname}-{bin_method}")
    
    plot_dap_maps(galname, bin_method, verbose=verbose)
    plot_local_maps(galname, bin_method, local_maps_path, verbose=verbose)
    plot_local_grid(galname, bin_method, local_maps_path, mask=True, verbose=verbose)
    velocity_vs_sfr(galname, bin_method, local_maps_path, pearson=True, contours=True, hists=True, verbose=verbose)
    terminal_velocity(galname, bin_method, local_maps_path, power_law=True, verbose=verbose)

    verbose_print(verbose, 'Done.')


if __name__ == "__main__":
    args = get_args()
    main(args)