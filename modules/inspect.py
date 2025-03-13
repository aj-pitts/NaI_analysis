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


from modules import defaults
from modules.util import verbose_print, verbose_warning, check_filepath

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def inspect_bin(galname, bin_method, bin_list, cube_file, map_file, redshift, local_file = None, show_norm = False, verbose = False, QtAgg = True, save = True):
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


def inspect_vel_ew(ewmap, vmap, snrmap, spatial_bins, fig_save_dir, static_sn_ranges = True, contour = True, thresholds = None, 
                   verbose = False):
        check_filepath(fig_save_dir, mkdir=True, verbose=verbose)
        plt.style.use(os.path.join(defaults.get_default_path(subdir='config'), 'figures.mplstyle'))

        out_file = os.path.join(fig_save_dir, 'vel_thresholds.pdf')
        out_file_plotly = os.path.join(fig_save_dir, 'vel_thresholds_binned.pdf')

        fig, ax = plt.subplots(2,2, sharey=True, sharex=False, figsize=(12,12))
        binarr = spatial_bins.flatten()
        snarr = snrmap.flatten()
        ewarr = ewmap.flatten()
        velarr = vmap.flatten()

        #snr_stats = (int(np.median(snarr) - 0.3 * np.std(snarr)), int(np.median(snarr)), int(np.median(snarr) + np.std(snarr)), np.ceil(np.max(snarr)))
        #snranges = [(0, snr_stats[0]), (snr_stats[0], snr_stats[1]), (snr_stats[1], snr_stats[2]), (snr_stats[2], snr_stats[3])]
        snranges = [(0, 30), (30, 60), (60, 90), (90, np.inf)]
        #thresholds = [None, 0.8, 0.6, 0.4]

        verbose_print(verbose, f"Creating velocity versus EW plots")

        for i, a in enumerate(fig.get_axes()):
            sn_range = snranges[i]
            w_sn = (snarr > sn_range[0]) & (snarr <= sn_range[1])
            w = (w_sn) & (ewarr!=-999) & (velarr !=-999)

            true_bins = np.unique(binarr[w])
            
            vels_cleaned = []
            ews_cleaned = []

            for bin_id in true_bins:
                binmask = (bin_id == binarr) & w
                vels_cleaned.append(np.median(velarr[binmask]))
                ews_cleaned.append(np.median(ewarr[binmask]))
            
            vels_cleaned = np.array(vels_cleaned)
            ews_cleaned = np.array(ews_cleaned)
            
            a.scatter(ews_cleaned, vels_cleaned, s=0.5)
            a.set_title(rf"${sn_range[0]} < S/N \leq {sn_range[1]}$")
            ewmin = np.floor(np.median(ews_cleaned) - 3 * np.std(ews_cleaned))
            ewmax = np.ceil(np.median(ews_cleaned) + 3 * np.std(ews_cleaned))
            a.set_xlim(ewmin,ewmax)
            a.set_ylim(-750, 750)

            a.grid(visible=True)
            if thresholds is not None:
                vline = thresholds[i]
                if vline is not None:
                    a.vlines(vline, -1000, 1000, linestyle='dashed', color='dimgrey', lw=1, label=rf'{vline} $\mathrm{{\AA}}$')
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
            for i, sn_range in enumerate(snranges):
                # Apply S/N range and data cleaning
                w_sn = (snarr > sn_range[0]) & (snarr <= sn_range[1])
                w = w_sn & (ewarr != -999) & (velarr != -999)

                true_bins = np.unique(binarr[w])
                
                vels_cleaned = []
                ews_cleaned = []

                for bin_id in true_bins:
                    binmask = (bin_id == binarr) & w
                    vels_cleaned.append(np.median(velarr[binmask]))
                    ews_cleaned.append(np.median(ewarr[binmask]))
                
                
                vels_cleaned = np.array(vels_cleaned)
                ews_cleaned = np.array(ews_cleaned)
                
                # Create 2D histogram for contour plot
                npoints = vels_cleaned.size
                points_per_bin = 75
                nbins = npoints/points_per_bin
                nbins = int(nbins) if nbins >= 1 else 1

                z = np.histogram2d(ews_cleaned, vels_cleaned, bins=nbins, range=[[-1, 3], [-750, 750]])[0]
                
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
                vline = None
                if vline is not None:
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
                    text=rf'$N_{{\mathrm{{points}}}} = {len(vels_cleaned)}$',
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




def velocity_vs_sfr(galname, bin_method, save_dir, terminal_velocity = True, power_law = True, pearson = True, contours = True, incidence = True,
                    hists = False, verbose = False):
    check_filepath(save_dir, mkdir=True, verbose=verbose)
    verbose_print(verbose, f"Creating velocity plots")

    ## Aquire the data and set up paths
    analysis_plan = defaults.analysis_plans()
    corr_key = 'BETA-CORR'
    local_data = defaults.get_data_path(subdir='local')

    filedir = os.path.join(local_data, 'local_outputs', f"{galname}-{bin_method}", corr_key, analysis_plan)
    filepath = os.path.join(filedir, f"{galname}-{bin_method}-local_maps.fits")

    hdul = fits.open(filepath)

    spatial_bins = hdul['SPATIAL_BINS'].data

    vmap = hdul['V_NaI'].data
    vmap_mask = hdul['V_NaI_MASK'].data
    vmap_error = np.median(hdul['V_NaI_ERROR'].data, axis=0)
    vmap_frac = hdul['V_NaI_FRAC'].data


    sfrmap = hdul['SFRSD'].data
    sfrmap_mask = hdul['SFRSD_MASk'].data
    sfrmap_error = hdul['SFRSD_ERROR'].data

    ### TODO HANDLE ASYMMETRIC ERRORS
    ## mask out values with map masks and velocity fracs
    mask = np.logical_or(vmap_mask.astype(bool), sfrmap_mask.astype(bool))

    mask_significant = np.logical_and(~mask, vmap_frac >= 0.95)
    mask_insignificant = np.logical_and(~mask, vmap_frac < 0.95)

    unique_bins, inds_sig = np.unique(spatial_bins[mask_significant], return_index=True)
    unique_bins, inds_insig = np.unique(spatial_bins[mask_insignificant], return_index=True)


    sfrs = sfrmap[mask_significant][inds_sig]
    sfr_errors = sfrmap_error[mask_significant][inds_sig]

    velocities = vmap[mask_significant][inds_sig]
    velocity_errors = vmap_error[mask_significant][inds_sig]

    ## colormap and errorbar style
    normalized = mcolors.Normalize(vmin = np.min(velocities), vmax = np.max(velocities))
    cmap = cm.bwr
    colors = cmap(normalized(velocities))

    style = dict(
        linestyle = 'none',
        marker = 'o',
        ms = 1.9,
        lw = 1.5,
        markeredgecolor = 'k',
        markeredgewidth = 0.6,
        ecolor = 'k',
        elinewidth = 0.6,
        capsize = 2
    )

    ## scatter for insignificant values, errorbar for significant values
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(sfrmap[mask_insignificant][inds_insig], vmap[mask_insignificant][inds_insig], s=0.5, color='k', alpha=0.7)
    for sf, vel, d_sfr, d_v, c in zip(sfrs, velocities, sfr_errors, velocity_errors, colors):
        errorbar = ax.errorbar(sf, vel, xerr=None, yerr=d_v, markerfacecolor=c, **style)
    
    ax.set_xlim(-2.6,-.1)
    ax.set_xlabel(r'$\mathrm{log\ \Sigma_{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}\ spaxel^{-1}} \right)$')
    ax.set_ylabel(r'$v_{\mathrm{Na\ D}}\ \left( \mathrm{km\ s^{-1}} \right)$')
    ax.set_ylim(-350, 350)

    ## pearson rank test
    if pearson:
        pearson_result = pearsonr(sfrs, velocities)
        ax.text(0.85, 0.85, fr'$\rho = {pearson_result[0]:.1f}$', fontsize=11, transform = ax.transAxes)


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

    outfil = os.path.join(save_dir, f'{galname}-{bin_method}-v_vs_sfr.pdf')
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
            
        outfil = os.path.join(save_dir, f'{galname}-{bin_method}-v_vs_sfr-contours.pdf')
        pio.write_image(fig, outfil) 
        verbose_print(verbose, f"Velocity vs SFR with contours saved to {outfil}")
    


    if terminal_velocity:
        vterm = hdul['V_MAX_OUT'].data
        vterm_mask = hdul['V_MAX_OUT_MASK'].data
        vterm_error = np.median(hdul['V_MAX_OUT_ERROR'].data, axis=0)

        mask = np.logical_or(vterm_mask.astype(bool), sfrmap_mask.astype(bool))

        terminal_velocities = vterm[~mask]
        terminal_velocity_errors = vterm_error[~mask]

        sfrs = sfrmap[~mask]
        sfr_errors = sfrmap_error[~mask]

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

        plt.figure(figsize=(7,7))
        for tv, tv_err, sfr, sfr_err, c in zip(terminal_velocities, terminal_velocity_errors, sfrs, sfr_errors, colors):
            #plt.errorbar(sfr, tv, xerr=None, yerr=tv_err, color = c, **style)
            plt.scatter(sfr, tv, marker='o', s = 4, color=c, ec='k', lw=1)

        ## setup power law and pearsonr
        if power_law:
            def wind_model(sfr, scale, power):
                return scale * sfr ** power
            
            popt, pcov = curve_fit(wind_model, 10**(sfrs), terminal_velocities, p0=(100, 0.1))

            modsfr = np.logspace(-3, 0, 1000)
            modv = wind_model(modsfr, popt[0], popt[1])

            plt.plot(np.log10(modsfr), modv, 'dimgray', linestyle='dashed', 
                    label=rf'$v = {popt[0]:.0f}\ \left( \Sigma_{{\mathrm{{SFR}}}} \right)^{{{popt[1]:.2f}}}$')

            plt.legend(frameon=False)

        plt.xlim(-2.4,0)
        plt.xlabel(r'$\mathrm{Log}\ \Sigma_{\mathrm{SFR}}\ \left( \mathrm{M_{\odot}\ yr^{-1}\ kpc^{-2}} \right)$')
        plt.ylabel(r'$ v_{\mathrm{out,\ max}}\ \left( \mathrm{km\ s^{-1}} \right)$')

        outfil = os.path.join(save_dir, f'{galname}-{bin_method}-terminalv_vs_sfr.pdf')
        plt.savefig(outfil, bbox_inches='tight')
        verbose_print(verbose, f"Terminal velocity vs SFR fig saved to {outfil}")

    if incidence:
        sfrs = sfrmap[mask_significant][inds_sig]
        sfr_errors = sfrmap_error[mask_significant][inds_sig]

        velocities = vmap[mask_significant][inds_sig]
        velocity_errors = vmap_error[mask_significant][inds_sig]

        # Freedman-Diaconis Rule
        iqr = np.percentile(sfrs, 75) - np.percentile(sfrs, 25)
        bin_width_fd = 2 * iqr / (len(sfrs) ** (1 / 3))
        bins_fd = int((np.ceil((sfrs.max() - sfrs.min()) / bin_width_fd))*1.5)

        bins = np.linspace(sfrs.min(), sfrs.max(), bins_fd)
        bin_indices = np.digitize(sfrs, bins)

        # Compute fractions
        f_in = []
        #F_in = []
        f_out = []
        #F_out = []
        for i in range(1, len(bins)):
            bin_mask = bin_indices == i
            total_count = np.sum(bin_mask)  # Total velocities in the bin

            ## inflow fraction
            positive_count = np.sum(velocities[bin_mask] > 0)  # Number of Velocities in the bin > 0
            fraction_inflow = positive_count / total_count if total_count > 0 else 0 # fraction of inflow relative to bin
            f_in.append(fraction_inflow)
            #F_in.append(positive_count/velocities.size) # fraction of inflow relative to full sample

            ## outflow fraction
            negative_count = np.sum(velocities[bin_mask] < 0) # Number of Velocities in the bin > 0
            fraction_outflow = negative_count / total_count if total_count > 0 else 0
            f_out.append(fraction_outflow)
            #F_out.append(negative_count/velocities.size)
        

        fig, axes = plt.subplots(1,2, figsize = (8,4), sharex=True)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Bin centers for plotting

        fraction_list = [f_in, f_out]
        color_list = ['tab:red', 'tab:blue']
        ylabels = [r"$f_{\mathrm{inflow,\ bin}}$", r"$f_{\mathrm{outflow,\ bin}}$", r"$F_{\mathrm{inflow,\ bin}}$", r"$F_{\mathrm{outflow,\ bin}}$"]
        #X = [0, 1] * 2
        for i, ax in enumerate(fig.get_axes()):
            #x = X[i]
            ax.bar(bin_centers, fraction_list[i], width = np.diff(bins), align='center', color = color_list[i], edgecolor = 'k', alpha = 0.8)
            ax.set_ylabel(ylabels[i], fontsize=14)

        fig.text(0.5, 0.05, r"$\mathrm{log\ \Sigma_{SFR}}$", ha='center', va='center', fontsize=18)
        fig.subplots_adjust(wspace=0.4)

        outfile = os.path.join(save_dir, f'{galname}-{bin_method}-incidence.pdf')
        plt.savefig(outfile, bbox_inches='tight')
        verbose_print(verbose, f"Incidence vs SFR fig saved to {outfil}")