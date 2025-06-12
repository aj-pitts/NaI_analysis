import os
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator

from modules import defaults, file_handler
from modules.util import verbose_print, verbose_warning, check_filepath

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from tqdm import tqdm


def inspect_bin(galname, bin_method, bin_list, cube_file, map_file, redshift, local_file = None, show_norm = False, verbose = False, QtAgg = True, save = True):
    if len(bin_list) > 100:
        raise ValueError(f"Input of {len(bin_list)} bins too large. Recommended 100 or less bins at a time.")
    
    plt.style.use(os.path.join(defaults.get_default_path('config'), 'figures.mplstyle'))
    if QtAgg:
        matplotlib.use('QtAgg')
    
    cube = fits.open(cube_file)
    maps = fits.open(map_file)

    spatial_bins = cube['binid'].data[0]
    flux = cube['flux'].data 
    wave = cube['wave'].data 
    stellar_cont = cube['model'].data 
    #ivar = cube['ivar'].data

    stellar_vel = maps['stellar_vel'].data 
     
    c = 2.998e5
    NaD_window = (5875, 5915)
    NaD_rest = [5891.5833, 5897.5581]

    if local_file is not None:
        local_maps = fits.open(local_file)
        try:
            mcmc_cube = local_maps['mcmc_results'].data
            mcmc_16 = local_maps['mcmc_16th_perc'].data 
            mcmc_84 = local_maps['mcmc_84th_perc'].data 
        except:
            mcmc_cube = None
            verbose_warning(verbose, f"Local data does not contain 'MCMC_RESULTS'")
    
    ndim = int(np.ceil(np.sqrt(len(bin_list))))
    ncol = min(ndim, 10)
    nrow = int(np.ceil(len(bin_list) / ncol))

    fig_width = ncol * 3
    fig_height = nrow * 3
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    fontsize = max(15, fig.get_size_inches()[1])

    total_subplots = len(bin_list)
    for i in range(total_subplots):
        row_idx = i // ncol  # Current row index
        col_idx = i % ncol   # Current column index
        # Calculate the number of subplots in the last row
        last_row_cols = total_subplots % ncol
        if last_row_cols == 0 and total_subplots != 0:
            last_row_cols = ncol  # If no remainder, the last row fills all columns


        Bin = bin_list[i]
        w = Bin == spatial_bins
        ny, nx = np.where(w)
        y = ny[0]
        x = nx[0]

        z = (stellar_vel[y, x] * (1 + redshift))/c + redshift
        
        restwave = wave / (1 + z)
        flux_1D = flux[:, y, x]
        model_1D = stellar_cont[:, y, x]
        #ivar_1D = ivar[:, y, x]

        wave_window = (restwave >= NaD_window[0]) & (restwave <= NaD_window[1])
        inds = np.where(wave_window)

        gs_parent = gridspec.GridSpec(nrow, ncol, figure=fig)
        subplot_spec = gs_parent[i]  # Get the SubplotSpec
        gs = subplot_spec.subgridspec(3, 1, height_ratios=[2,1,0], hspace=0)
        #gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=fig.add_subplot(nrow, ncol, i+1), height_ratios=[3,1])

        ax1 = fig.add_subplot(gs[0])

        ax1.plot(restwave[inds], flux_1D[inds] / model_1D[inds], 'k', drawstyle='steps-mid')

        #ax.set_title(f"Bin {Bin}")
        ax1.text(.1,.85, f"Bin {Bin}", transform=ax1.transAxes)

        if mcmc_cube is not None:
            lambda_0 = mcmc_cube[0, y, x]
            lambda_16 = mcmc_16[0, y, x]
            lambda_84 = mcmc_84[0, y, x]

            ax1.fill_between([lambda_0-lambda_16, lambda_0+lambda_84], [-20, -20], [20, 20], color='r',
                            alpha=0.3)
            ax1.vlines([lambda_0], -20, 20, colors = 'black', linestyles = 'dashed', linewidths = .6)

        ax1.vlines(NaD_rest, -20, 20, colors = 'dimgray', linestyles = 'dotted', linewidths = .5)
        
        ax1.set_ylim(.85, 1.15)
        ax1.set_xlim(NaD_window[0], NaD_window[1])
        ax1.set_box_aspect(3/4)




        ax2 = fig.add_subplot(gs[1], sharex = ax1)

        min_flux = round(min(flux_1D[inds].min(), model_1D[inds].min()), 2)
        max_flux = round(max(flux_1D[inds].max(), model_1D[inds].max()), 2)
        med_flux = round(max(np.median(flux_1D[inds]), np.median(model_1D[inds])), 2)

        ax2.plot(restwave[inds], flux_1D[inds] / med_flux, 'dimgray', drawstyle = 'steps-mid', lw=1.4)
        ax2.plot(restwave[inds], model_1D[inds] / med_flux, 'tab:blue', drawstyle = 'steps-mid', lw=1.1)

        
        ax2.set_ylim(1 - 0.3, 1 + 0.2)
        ax2.set_box_aspect(3/8)

        if col_idx >= 1:
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])

        if row_idx < nrow - 2:
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
        if row_idx == nrow - 2 and col_idx < last_row_cols:
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])

    fig.text(0.5, 0.05, r'Wavelength $\left( \mathrm{\AA} \right)$', ha='center', va='center', fontsize=fontsize)
    # fig.text(0.05, 0.5, r'Flux (top: Normalized, Bottom: $\left[ \mathrm{1E-17\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}\ spaxel^{-1}} \right]$)',
    #          rotation='vertical',ha='center',va='center', fontsize=fontsize)
    fig.text(0.05, 0.5, r'Normalized Flux', rotation='vertical',ha='center',va='center', fontsize=fontsize)
    #fig.text(0.06, 0.5, r'top: Normalized', rotation='vertical',ha='center',va='center', fontsize=fontsize)
    #fig.text(0.07, 0.5, r'bottom: $\mathrm{1E-17\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}\ spaxel^{-1}}$', rotation='vertical',ha='center',va='center', fontsize=fontsize)

    fig.subplots_adjust(wspace=0.01, hspace=0.05)
    if QtAgg:
        plt.show(block=False)
    else:
        local_data = defaults.get_data_path('local')
        output_dir = os.path.join(local_data, 'local_outputs', f"{galname}-{bin_method}", "figures")
        if not os.path.exists(output_dir):
            raise ValueError(f"Cannot save figure; Path does not exist: {output_dir}")
        if save:
            plt.savefig(os.path.join(output_dir, 'Bin_Inspect.pdf'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def inspect_vel_ew(galname, bin_method, ewmap, ewmap_mask, snrmap, vmap, spatial_bins, contour = True, fig_save_dir = None,
                   verbose = False):
        thresholds = file_handler.threshold_parser(galname, bin_method, require_ew=False)
        snranges = thresholds['sn_lims']
        ewlims = thresholds['ew_lims']

        if fig_save_dir is not None:
            check_filepath(fig_save_dir, mkdir=False, verbose=verbose)
        plt.style.use(os.path.join(defaults.get_default_path(subdir='config'), 'figures.mplstyle'))

        outpath = defaults.get_fig_paths(galname, bin_method, 'inspection') if fig_save_dir is None else fig_save_dir
        out_file = os.path.join(outpath, 'vel_thresholds.pdf')
        out_file_plotly = os.path.join(outpath, 'vel_thresholds_binned.pdf')

        w = (vmap == -999) | (ewmap == -999)
        mask = np.logical_or(ewmap_mask.astype(bool), w)

        masked_bins = spatial_bins[~mask]
        masked_snr = snrmap[~mask]
        masked_ew = ewmap[~mask]
        masked_vel = vmap[~mask]

        _, bin_inds = np.unique(masked_bins, return_index=True)
        
        snarr = masked_snr[bin_inds]
        ewarr = masked_ew[bin_inds]
        velarr = masked_vel[bin_inds]


        verbose_print(verbose, f"Creating velocity versus EW plots")

        fig, ax = plt.subplots(2,2, sharey=True, sharex=False, figsize=(12,12))
        for i, a in enumerate(fig.get_axes()):
            sn_low, sn_high = snranges[i]
            w = (snarr > sn_low) & (snarr <= sn_high)
            
            a.scatter(ewarr[w], velarr[w], s=0.5)
            a.set_title(rf"${sn_low} < S/N \leq {sn_high}$")

            ewmin = np.floor(np.median(ewarr[w]) - 3 * np.std(ewarr[w]))
            ewmax = np.ceil(np.median(ewarr[w]) + 3 * np.std(ewarr[w]))
            a.set_xlim(ewmin,ewmax)
            a.set_ylim(-750, 750)

            a.grid(visible=True, linewidth=0.5, zorder=0)
            if ewlims is not None:
                vline = ewlims[i]
                if not np.isfinite(vline):
                    pass
                a.vlines(vline, -1000, 1000, linestyle='dashed', color='k', lw=1.5, label=rf'{vline} $\mathrm{{\AA}}$')
                a.legend(frameon=False, loc='upper right')
            
        fig.text(0.5, 0, r'$\mathrm{EW_{Na\ D}}\ (\mathrm{\AA})$', ha='center',va='center',fontsize=20)
        fig.text(0, 0.5, r'$v_{\mathrm{Na\ D}}\ (\mathrm{km\ s^{-1}})$', rotation='vertical', ha='center', va='center', fontsize=20)
        plt.savefig(out_file,bbox_inches='tight')

        verbose_print(verbose, f"Scatter plot saved to {out_file}")


        if contour:
            # Create subplots
            fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"${sn_range[0]} < S/N ≤ {sn_range[1]}$" for sn_range in snranges],
                                horizontal_spacing=0.075, vertical_spacing=0.075,
                            x_title = r'$\mathrm{EW_{Na\ D}\ (Å)}$', y_title = r'$v_{\mathrm{Na\ D}}\ \mathrm{(km\ s^{-1})}$')


            # Iterate through S/N ranges and create contours
            for i, (sn_low, sn_high) in enumerate(snranges):
                # Apply S/N range and data cleaning
                w = (snarr > sn_low) & (snarr <= sn_high)

                # Create 2D histogram for contour plot
                npoints = velarr[w].size
                points_per_bin = 75
                nbins = npoints/points_per_bin
                nbins = int(nbins) if nbins >= 1 else 1

                z = np.histogram2d(ewarr[w], velarr[w], bins=nbins, range=[[-1, 3], [-750, 750]])[0]
                
                verbose_print(verbose, f"Plotting contours with {int(nbins)} bins")

                fig.add_trace(
                    go.Contour(
                        z=z.T,  # Transpose to match the orientation of axes
                        x=np.linspace(-1, 3, z.shape[1]),
                        y=np.linspace(-750, 750, z.shape[0]),
                        colorscale='Greys',
                        showscale=False
                    ),
                    row=(i//2)+1, col=(i%2)+1
                )

                # Add vertical line if threshold exists
                #vline = thresholds[i]
                vline = ewlims[i]
                if np.isfinite(vline):
                    fig.add_trace(
                        go.Scatter(
                            x=[vline, vline],
                            y=[-1000, 1000],
                            mode='lines',
                            line=dict(dash='dash', color='dimgrey', width=1),
                            showlegend=False
                            #name=f'{vline} Å'  # Removed LaTeX inside name for non-math mode
                        ),
                        row=(i//2)+1, col=(i%2)+1
                    )
                    # Add annotation next to the vertical line
                    fig.add_annotation(
                        x=vline+0.2, y=600,
                        text=f'{vline} Å',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        ax=10,  # Adjust horizontal distance
                        ay=0,   # Adjust vertical distance
                        row=(i//2)+1, col=(i%2)+1
                    )
                fig.add_annotation(
                    x=1.6, y=300,
                    text=rf'$N_{{\mathrm{{points}}}} = {len(velarr[w])}$',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    ax=10,  # Adjust horizontal distance
                    ay=0,   # Adjust vertical distance
                    row=(i//2)+1, col=(i%2)+1
                )


            # Update layout with titles and axis labels
            fig.update_layout(
                height=800, width=800,
                #showlegend=True,
            )

            # Update axes for each subplot (add axis lines and tick marks)
            for i in range(1, 5):
                fig.update_xaxes(
                    range=[-1, 2.5],
                    linecolor='black',
                    linewidth=2,
                    showline=True,
                    showgrid=True,
                    ticks="inside",
                    ticklen=8,
                    tickmode="array",
                    minor_ticks='inside', 
                    minor_ticklen=3,
                    row=(i-1)//2+1, col=(i-1)%2+1,
                    mirror='all',
                )
                
                fig.update_yaxes(
                    range=[-750, 750],
                    linecolor='black',
                    linewidth=2,
                    showline=True,
                    showgrid=True,
                    ticks="inside",
                    ticklen=8,
                    tickmode="array",
                    minor_ticks='inside',
                    minor_ticklen=3,
                    row=(i-1)//2+1, col=(i-1)%2+1,
                    mirror='all',
                )
                
            pio.write_image(fig, out_file_plotly) 
            verbose_print(verbose, f"Contour plot saved to {out_file_plotly}")