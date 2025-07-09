import os
import numpy as np
from astropy.io import fits

from glob import glob

import argparse

import matplotlib.pyplot as plt
from matplotlib import gridspec

from modules import defaults, file_handler
from modules.util import verbose_print, verbose_warning, check_filepath

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def threshold_bins(galname, bin_method, nbins = 10, verbose = False):
    datapath_dict = file_handler.init_datapaths(galname, bin_method, verbose = False)
    local_file = datapath_dict['LOCAL']
    hdul = fits.open(local_file)

    spatial_bins = hdul['spatial_bins'].data.flatten()
    unique_bins, bin_inds = np.unique(spatial_bins, return_index=True)

    snr = hdul['nai_snr'].data.flatten()[bin_inds]
    ew = hdul['ew_nai'].data.flatten()[bin_inds]
    ew_mask = hdul['ew_nai_mask'].data.flatten().astype(bool)[bin_inds]

    threshold_dict = file_handler.threshold_parser(galname, bin_method, require_ew=True)
    bin_dict = {}

    for (sn_low, sn_high), ew_lim in zip(threshold_dict['sn_lims'], threshold_dict['ew_lims']):
        if not np.isfinite(ew_lim):
            continue
        sn_low = int(sn_low) if np.isfinite(sn_low) else sn_low
        sn_high = int(sn_high) if np.isfinite(sn_high) else sn_high
        key = f'{sn_low}_{sn_high}'
        bin_dict[key] = {'below':{}, 'above':{}}

        w = (snr > sn_low) & (snr <= sn_high) & (ew < ew_lim) & (ew >= ew_lim - 0.1)
        mask = np.logical_and(w, ~ew_mask)
        masked_ew = ew[mask]
        masked_bins = unique_bins[mask]
        masked_snr = snr[mask]

        sort_inds = np.argsort(-masked_ew)
        sorted_ew = masked_ew[sort_inds]
        sorted_bins = masked_bins[sort_inds]
        sorted_snr = masked_snr[sort_inds]
        bin_dict[key]['below']['ew'] = sorted_ew
        bin_dict[key]['below']['snr'] = sorted_snr
        bin_dict[key]['below']['bins'] = sorted_bins


        w = (snr > sn_low) & (snr <= sn_high) & (ew >= ew_lim) & (ew < ew_lim + 0.1)
        mask = np.logical_and(w, ~ew_mask)
        masked_ew = ew[mask]
        masked_bins = unique_bins[mask]
        masked_snr = snr[mask]

        sort_inds = np.argsort(-masked_ew)
        sorted_ew = masked_ew[sort_inds]
        sorted_bins = masked_bins[sort_inds]
        sorted_snr = masked_snr[sort_inds]
        bin_dict[key]['above']['ew'] = sorted_ew
        bin_dict[key]['above']['snr'] = sorted_snr
        bin_dict[key]['above']['bins'] = sorted_bins

    for key in bin_dict.keys():
        subdict = bin_dict[key]

        inspect_bin_profiles(galname, bin_method, subdict['below']['bins'][:nbins], subdict['below']['snr'][:nbins], subdict['below']['ew'][:nbins], show = False, save = True, 
                             fname = f'{galname}-{bin_method}_{key}_badbin_inspect.pdf', verbose=verbose)
        inspect_bin_profiles(galname, bin_method, subdict['above']['bins'][:nbins], subdict['above']['snr'][:nbins], subdict['above']['ew'][:nbins], show = False, save = True, 
                             fname = f'{galname}-{bin_method}_{key}_goodbin_inspect.pdf', verbose=verbose)

def inspect_bin_profiles(galname, bin_method, bin_list, snrs = None, ews = None,
                         show = False, save = True, fname = None, verbose = False):
    if len(bin_list) > 100:
        raise ValueError(f"Input of {len(bin_list)} bins too large. Recommended 100 or less bins at a time.")
    
    if len(bin_list) != len(snrs):
        raise ValueError(f"S/N array does not match Bin array")
    if len(bin_list) != len(ews):
        raise ValueError(f"EW array does not match Bin array")
    
    verbose_print(f"Plotting line profiles for {len(bin_list)} bins...")
    plt.style.use(os.path.join(defaults.get_default_path('config'), 'figures.mplstyle'))

    datapath_dict = file_handler.init_datapaths(galname, bin_method, verbose = False)
    local_file = datapath_dict['LOCAL']
    cube_file = datapath_dict['LOGCUBE']
    
    cube = fits.open(cube_file)
    local_maps = fits.open(local_file)

    spatial_bins = cube['binid'].data[0]
    flux = cube['flux'].data 
    wave = cube['wave'].data 
    stellar_cont = cube['model'].data 
    #ivar = cube['ivar'].data

    redshift = local_maps['redshift'].data
    mcmc_cube = local_maps['mcmc_results'].data
    mcmc_16 = local_maps['mcmc_16th_perc'].data 
    mcmc_84 = local_maps['mcmc_84th_perc'].data 
    
    NaD_window = (5875, 5915)
    #NaD_window = (5880, 5920)
    NaD_rest = [5891.5833, 5897.5581]

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
        is_last_row = (row_idx == nrow - 1)
        last_row_cols = total_subplots % ncol
        if last_row_cols == 0 and total_subplots != 0:
            last_row_cols = ncol  # If no remainder, the last row fills all columns

        Bin = bin_list[i]
        sn = snrs[i]
        ew = ews[i]
        w = Bin == spatial_bins
        ny, nx = np.where(w)
        y = ny[0]
        x = nx[0]

        z = redshift[y, x]
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
        ax1.hlines([1], xmin=NaD_window[0], xmax=NaD_window[1], colors='k', alpha=0.4, linewidths=0.5, linestyles='dashed')

        #ax.set_title(f"Bin {Bin}")
        ax1.text(.075,.85, f"Bin {Bin}", transform=ax1.transAxes)
        ax1.text(.925,.725, rf"$S/N = {sn:.0f}$", transform=ax1.transAxes, ha='right')
        ax1.text(.925,.85, rf"$\mathrm{{EW}} = {ew:.4f}\ \mathrm{{\AA}}$", transform=ax1.transAxes, ha='right')

        lambda_0 = mcmc_cube[0, y, x]
        lambda_16 = mcmc_16[0, y, x]
        lambda_84 = mcmc_84[0, y, x]

        ax1.fill_between([lambda_0-lambda_16, lambda_0+lambda_84], [-20, -20], [20, 20], color='r',
                        alpha=0.3)
        ax1.vlines([lambda_0], -20, 20, colors = 'black', linestyles = 'dashed', linewidths = 1)

        ax1.vlines(NaD_rest, -20, 20, colors = 'black', linestyles = 'dotted', linewidths = .8)
        
        ax1.set_ylim(.75, 1.2)
        ax1.set_xlim(NaD_window[0], NaD_window[1])
        ax1.set_box_aspect(3/4)
        ax1.set_xticklabels([])


        ax2 = fig.add_subplot(gs[1])

        # min_flux = round(min(flux_1D[inds].min(), model_1D[inds].min()), 2)
        # max_flux = round(max(flux_1D[inds].max(), model_1D[inds].max()), 2)
        med_flux = round(max(np.median(flux_1D[inds]), np.median(model_1D[inds])), 2)

        ax2.plot(restwave[inds], flux_1D[inds] / med_flux, 'dimgray', drawstyle = 'steps-mid', lw=1.4)
        ax2.plot(restwave[inds], model_1D[inds] / med_flux, 'tab:blue', drawstyle = 'steps-mid', lw=1.1)
        ax2.vlines(NaD_rest, -20, 20, colors = 'black', linestyles = 'dotted', linewidths = .8)
        
        ax2.set_ylim(.65, 1.2)
        ax2.set_xlim(NaD_window[0], NaD_window[1])
        ax2.set_box_aspect(3/8)

        if col_idx >= 1:
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])

        # Remove x tick labels on all rows except the last one
        if not is_last_row:
            ax2.set_xticklabels([])

        # (Optional) Handle second-to-last row cleanup only when last row is partial
        if last_row_cols != ncol:
            if row_idx == nrow - 2 and col_idx >= last_row_cols:
                ax2.set_xticklabels([])

    fig.text(0.5, 0.05, r'Wavelength $\left( \mathrm{\AA} \right)$', ha='center', va='center', fontsize=fontsize)
    # fig.text(0.05, 0.5, r'Flux (top: Normalized, Bottom: $\left[ \mathrm{1E-17\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}\ spaxel^{-1}} \right]$)',
    #          rotation='vertical',ha='center',va='center', fontsize=fontsize)
    fig.text(0.05, 0.5, r'Normalized Flux', rotation='vertical',ha='center',va='center', fontsize=fontsize)
    #fig.text(0.06, 0.5, r'top: Normalized', rotation='vertical',ha='center',va='center', fontsize=fontsize)
    #fig.text(0.07, 0.5, r'bottom: $\mathrm{1E-17\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}\ spaxel^{-1}}$', rotation='vertical',ha='center',va='center', fontsize=fontsize)

    fig.subplots_adjust(wspace=-0.15, hspace=0.05)
    if save:
        output_dir = defaults.get_fig_paths(galname, bin_method, subdir = 'inspection')
        if fname is None:
            fname = 'bin_inspect.pdf'
        else:
            root, ext = os.path.splitext(fname)
            if ext != '.pdf':
                verbose_warning(verbose, f"Filename extension of {ext} invalid. Defaulting to PDF")
                fname = root+'.pdf'

        figpath = os.path.join(output_dir, fname)
        plt.savefig(figpath, bbox_inches='tight')
        verbose_print(verbose, f'Bin inspection figure saved to {figpath}')
    if show:
        plt.show()
    else:
        plt.close()


def inspect_vstd_ew(galname, bin_method, threshold_data, vmap, vmap_error, vmap_mask = None, 
                    ewnoem = False, scatter_lim = 30, fig_save_dir = None, verbose=False):
    if fig_save_dir is not None:
        check_filepath(fig_save_dir, mkdir=False, verbose=verbose)
    plt.style.use(os.path.join(defaults.get_default_path(subdir='config'), 'figures.mplstyle'))

    outpath = defaults.get_fig_paths(galname, bin_method, 'inspection') if fig_save_dir is None else fig_save_dir

    file_end = '_maskedem' if ewnoem else ''
    out_file = os.path.join(outpath, f'{galname}-{bin_method}_v_scatter{file_end}.pdf')

    thresholds = file_handler.threshold_parser(galname, bin_method, require_ew=False)
    snranges = thresholds['sn_lims']
    ewlims = thresholds['ew_lims']

    datapath_dict = file_handler.init_datapaths(galname, bin_method, verbose=False, redshift=False)
    local_file = datapath_dict['LOCAL']
    hdul = fits.open(local_file)

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

    fig = plt.figure(figsize=(8,8))

    for i, (sn_low, sn_high) in enumerate(snranges):
        sn_low = int(sn_low) if np.isfinite(sn_low) else sn_low
        sn_high = int(sn_high) if np.isfinite(sn_high) else sn_high

        gs_parent = gridspec.GridSpec(2, 2, figure=fig)
        subplot_spec = gs_parent[i]  # Get the SubplotSpec
        gs = subplot_spec.subgridspec(2, 1, height_ratios=[2,1], hspace=0.1)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        w = (snr > sn_low) & (snr <= sn_high)
        mask = np.logical_and(w, ~datamask)
        
        ew_bin = ew[mask]
        v_bin = velocity[mask]
        
        ax1.scatter(ew_bin, v_bin, s=0.5)
        ax1.vlines(ewlims[i], -1000, 1000, colors='k', linestyles='dotted', linewidths=1.5, label=rf'{ewlims[i]:.1f} $\mathrm{{\AA}}$')
        ax1.legend(frameon=False, loc='upper right')
        ax1.grid(visible=True, linewidth=0.5, zorder=0)

        ax1.set_ylim(-750, 750)
        ax1.set_xlim(-1, 3)
        ax1.set_xticklabels([])
        ax1.tick_params(labelsize=12)

        ax1.set_title(rf"${sn_low} < S/N \leq {sn_high}$")
        ax1.set_ylabel(r'$v_{\mathrm{cen}}\ \mathrm{(km\ s^{-1})}$',fontsize=14)


        key = f"{sn_low}-{sn_high}"
        subdict = threshold_data[key]
        v_std = subdict['std']
        med_ew = subdict['medew']
        ax2.plot(med_ew, v_std, drawstyle='steps-mid', color='dimgray')
        ax2.set_yscale('log')
        ax2.hlines([scatter_lim], -10, 10, colors='k', linestyles='dashed', linewidths = 1)

        ax2.set_ylim(0, 1000)
        ax2.set_xlim(-1, 3)
        ax2.tick_params(labelsize=12)

        ax2.set_ylabel(r'$\mathrm{med}\ \sigma_{v_{\mathrm{cen}}}$', fontsize=14)
        if i == 1 or i == 3:
            ax1.set_ylabel('')
            ax2.set_ylabel('')

    fig.text(0.5, 0, r'$\mathrm{EW_{Na\ D}}\ (\mathrm{\AA})$', ha='center',va='center',fontsize=18)
    plt.savefig(out_file, bbox_inches='tight')
    verbose_print(verbose, f"Saving velocity scatter figure to {out_file}")




def inspect_vel_ew(galname, bin_method, contour = True, fig_save_dir = None,
                   verbose = False):
    if fig_save_dir is not None:
        check_filepath(fig_save_dir, mkdir=False, verbose=verbose)
    outpath = defaults.get_fig_paths(galname, bin_method, 'inspection') if fig_save_dir is None else fig_save_dir
    out_file = os.path.join(outpath, 'vel_thresholds.pdf')
    out_file_plotly = os.path.join(outpath, 'vel_thresholds_binned.pdf')

    plt.style.use(os.path.join(defaults.get_default_path(subdir='config'), 'figures.mplstyle'))

    thresholds = file_handler.threshold_parser(galname, bin_method, require_ew=False)
    snranges = thresholds['sn_lims']
    ewlims = thresholds['ew_lims']

    datapath_dict = file_handler.init_datapaths(galname, bin_method, verbose=False, redshift=False)
    local_file = datapath_dict['LOCAL']
    hdul = fits.open(local_file)

    spatial_bins = hdul['spatial_bins'].data.flatten()
    unique_bins, bin_inds = np.unique(spatial_bins, return_index=True)

    vmap = hdul['v_nai'].data.flatten()[bin_inds]
    vmap_mask = hdul['v_nai_mask'].data.flatten()[bin_inds]
    vmap_mask[vmap_mask==12] = 0
    vmap_mask = vmap_mask.astype(bool)

    ewmap = hdul['ew_nai'].data.flatten()[bin_inds]
    ewmap_mask = hdul['ew_nai_mask'].data.flatten().astype(bool)[bin_inds]
    snrmap = hdul['nai_snr'].data.flatten()[bin_inds]

    mask = np.logical_and(ewmap_mask, vmap_mask)

    masked_snr = snrmap[~mask]
    masked_ew = ewmap[~mask]
    masked_vel = vmap[~mask]
    

    verbose_print(verbose, f"Creating velocity versus EW plots")

    fig, ax = plt.subplots(2,2, sharey=True, sharex=False, figsize=(12,12))
    for i, a in enumerate(fig.get_axes()):
        sn_low, sn_high = snranges[i]
        w = (masked_snr > sn_low) & (masked_snr <= sn_high)
        
        a.scatter(masked_ew[w], masked_vel[w], s=0.5)
        a.set_title(rf"${sn_low} < S/N \leq {sn_high}$")

        ewmin = np.floor(np.median(masked_ew[w]) - 3 * np.std(masked_ew[w]))
        ewmax = np.ceil(np.median(masked_ew[w]) + 3 * np.std(masked_ew[w]))
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
    fig.text(0, 0.5, r'$v_{\mathrm{cen}}\ (\mathrm{km\ s^{-1}})$', rotation='vertical', ha='center', va='center', fontsize=20)
    plt.savefig(out_file,bbox_inches='tight')

    verbose_print(verbose, f"Scatter plot saved to {out_file}")


    if contour:
        # Create subplots
        fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"${sn_range[0]} < S/N ≤ {sn_range[1]}$" for sn_range in snranges],
                            horizontal_spacing=0.075, vertical_spacing=0.075,
                        x_title = r'$\mathrm{EW_{Na\ D}\ (Å)}$', y_title = r'$v_{\mathrm{cen}}\ \mathrm{(km\ s^{-1})}$')


        # Iterate through S/N ranges and create contours
        for i, (sn_low, sn_high) in enumerate(snranges):
            # Apply S/N range and data cleaning
            w = (masked_snr > sn_low) & (masked_snr <= sn_high)

            # Create 2D histogram for contour plot
            npoints = masked_vel[w].size
            points_per_bin = 75
            nbins = npoints/points_per_bin
            nbins = int(nbins) if nbins >= 1 else 1

            z = np.histogram2d(masked_ew[w], masked_vel[w], bins=nbins, range=[[-1, 3], [-750, 750]])[0]
            
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
                text=rf'$N_{{\mathrm{{points}}}} = {len(masked_vel[w])}$',
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

def inspect_v_vs_r(galname, bin_method, verbose = False):

    return


def get_args():
    parser = argparse.ArgumentParser(description="A script to create/overwrite plots without rerunning analyze_NaI")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('nbins', type=int, help="Number of bins to plot in each figure (default: 10)", default=10)
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)

    return parser.parse_args()


def main(args):
    galname = args.galname
    bin_method = args.bin_method
    nbins = args.nbins
    verbose = args.verbose

    threshold_bins(galname, bin_method, nbins=nbins, verbose=verbose)


if __name__ == '__main__':
    args = get_args()
    main(args)