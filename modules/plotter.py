import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from mpl_toolkits.axes_grid1 import make_axes_locatable
import modules.defaults as defaults
import modules.util as util
import modules.inspect as inspect
from modules.util import verbose_print

import bokeh.plotting as bp
from bokeh.models import ColumnDataSource, TapTool, CustomJS, ColorBar, LinearColorMapper, Div
from bokeh.layouts import column
from bokeh.transform import linear_cmap
from bokeh.io import output_file, show, save
from bokeh.models import BasicTicker, ColorBar


def standard_plotting(galname, bin_method, corr_key, mapsfile, verbose=False):
    analysis_plans = defaults.analysis_plans()

    local_data = defaults.get_data_path(subdir='local')
    local_outputs = os.path.join(local_data, 'local_outputs', f"{galname}-{bin_method}", corr_key, analysis_plans)
    local_maps_path = os.path.join(local_outputs, f"{galname}-{bin_method}-local_maps.fits")

    figures_dir = os.path.join(local_outputs, 'figures')
    results_dir = os.path.join(figures_dir, 'results')
    dap_figures_dir = os.path.join(figures_dir, 'dap')

    hdul = fits.open(local_maps_path)

    util.verbose_print(verbose, "Creating plots...")

    map_plotter(galname, bin_method, hdul['nai_snr'].data, 'NAI_SNR', results_dir, r'$S/N_{\mathrm{Na\ D}}$', '', 'managua',
                0, 100, histogram=True, verbose=verbose)

    map_plotter(galname, bin_method, hdul['ew_nai'].data, 'EW_NAI', results_dir, r'$\mathrm{EW_{Na\ D}}$', r'$\left( \mathrm{\AA} \right)$', 'rainbow',
                -0.2, 1.5, mask = hdul['ew_nai_mask'].data, histogram=True, verbose=verbose)
    
    map_plotter(galname, bin_method, hdul['sfrsd'].data, 'SFRSD', results_dir, r"$\mathrm{log \Sigma_{SFR}}$", r"$\left( \mathrm{M_{\odot}\ kpc^{-2}\ yr^{-1}\ spaxel^{-1}} \right)$",
                'rainbow', -2.5, 0, mask = hdul['sfrsd_mask'].data, histogram=True, verbose=verbose)
    
    map_plotter(galname, bin_method, hdul['v_nai'].data, 'V_NaI_masked', results_dir, r"$v_{\mathrm{Na\ D}}$", r"$\left( \mathrm{km\ s^{-1}} \right)$", 'seismic', 
                -250, 250, mask=hdul['v_nai_mask'].data, minmax=True, mask_ignore_list=[8], histogram=True, verbose=verbose)
    
    map_plotter(galname, bin_method, hdul['v_nai'].data, 'V_NaI', results_dir, r"$v_{\mathrm{Na\ D}}$", r"$\left( \mathrm{km\ s^{-1}} \right)$", 'seismic', 
                -200, 200, minmax=True, histogram=True, verbose=verbose)

    inspect.dap_maps(galname, bin_method, mapsfile, dap_figures_dir, verbose = verbose)




def map_plotter(galname, bin_method, image, fig_keyword, save_path, label, units, cmap, vmin = None, vmax = None,
                mask = None, mask_ignore_list = None, minmax = False, histogram = True, verbose = True):
    
    image_name = f'{galname}-{bin_method}-{fig_keyword}.pdf'
    util.check_filepath(save_path,mkdir=True,verbose=verbose)
    fig_path = os.path.join(save_path, image_name)

    plt.style.use(defaults.matplotlib_rc())
    
    ####### 2D Map #########
    image_mask = np.zeros(image.shape).astype(bool)
    if mask is not None:
        image_mask = mask.astype(bool)

        if mask_ignore_list is not None:
            for i in mask_ignore_list:
                image_mask[image_mask == i] = 0

    boundary_mask = np.zeros(image.shape).astype(bool)
    if minmax:
        boundary_mask = (image<vmin) | (image>vmax)
    
    plotmap = np.copy(image)
    plot_mask = np.logical_or(image_mask, boundary_mask)
    plotmap[plot_mask] = np.nan

    value_string = f"{label} {units}"

    plt.figure()
    im = plt.imshow(plotmap,cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',
               extent=[32.4, -32.6,-32.4, 32.6])
    plt.xlabel(r'$\Delta \alpha$ (arcsec)')
    plt.ylabel(r'$\Delta \delta$ (arcsec)')
    ax = plt.gca()
    ax.set_facecolor('lightgray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.01)

    cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
    cbar.set_label(value_string,fontsize=15,labelpad=-55)
    cax.xaxis.set_ticks_position('top')
    

    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    verbose_print(verbose, f"{galname} {fig_keyword} map plot saved to {fig_path}")


    ####### Histogram #######
    if histogram:
        hist_name = f'{galname}-{bin_method}-{fig_keyword}-hist.pdf'
        hist_path = os.path.join(save_path, hist_name)

        masked_data = image[~image_mask]
        # median = np.median(masked_data)
        # standard_deviation = np.std(masked_data)

        # bin_width = 3.5 * standard_deviation / (masked_data.size ** (1/3))
        # nbins = (max(masked_data) - min(masked_data)) / bin_width
        # nbins = len(masked_data)//100 if not np.isfinite(nbins) else nbins
        nbins = 100
        plt.figure()
        plt.hist(masked_data,bins=int(nbins),color='k')
        plt.xlabel(label)
        plt.ylabel(r"$N_{\mathrm{bins}}$")
        #plt.xlim(median - 7 * standard_deviation, median + 7 * standard_deviation)

        plt.savefig(hist_path, bbox_inches='tight')
        plt.close()
        print(f"{galname} {fig_keyword} histogram saved to {hist_path}")



def map_plotter_old(image: np.ndarray, mask: np.ndarray, fig_save_path: str, fig_keyword: str, label: str, units: str, galname: str, 
             bin_method: str, ignore_mask_list = None, verbose = True, **kwargs):
    """
    Creates the set of 2D "maps" for a given measurement distribution using `matplotlib.pyplot`.

    Using the 2D numpy arrays `image` and `mask` creates map plots with `matplotlib.pyplot.imshow`.
    The default, guaranteed, map plot is data range limited by \\(s = 3\\) standard deviations above and
    below the median of the masked values of `image`. Optionally, `vmin` and/or `vmax` may be input
    where a new plot will be created with the specified limits. Also plots a histogram of the
    distribution of `image` values. All figures are saved as PDFs by default; if another file type
    is specified with `figext`, it is recommended to specify the figure DPI with the `dpi` kwarg.
    If the map of propagated uncertainties is included as `error`, an additional map of the
    uncertainties will be plotted.

    Parameters
    ----------
    image : np.ndarray
        ...

    mask : np.ndarray
        Boolean or integer mask corresponding to `image`.

    fig_keyword : str
        ...

    label : str
        The string of the label of the value to be passed into `matplotlib.pyplot` label methods 
        and kwargs.

    units : str
        The string of the units of the value to be passed into `matplotlib.pyplot` label methods
        and kwargs. Recommended to use LaTeX math mode format.

    verbose : bool, optional
        ...

    **kwargs : keyword arguments, optional
        Optional keyword arguments representing additional data. The following keyword arguments 
        are accepted and recommended. Any argument which can be passed into 
        `matplotlib.pyplot.imshow` is also accepted:

        | Keyword     | Type       | Description                                                  |
        |-------------|------------|--------------------------------------------------------------|
        | std         | float      | The number of standard deviations above and below the        |
        |             |            | median to restrict the colorscale of the "std plot". If not  |
        |             |            | specified, the default is 3.                                 |
        | figname     | str        | The leading name of each figure file to be written.          |
        | figext      | str        | The filetype extension of the figure file.                   |
        | error       | np.ndarray | Map of propagated uncertainties corresponding to `image`.    |
        
    """
    ### init verbose logger

    plt.style.use(defaults.matplotlib_rc())

    ### unpack and handle defaults for all kwargs
    # names and paths
    figext = kwargs.get('figext', 'pdf')
    figname = kwargs.get('figname', f'{galname}-{bin_method}-{fig_keyword}')

    histfigname = os.path.join(fig_save_path, f"{figname}-histogram.{figext}")
    stdfigname = os.path.join(fig_save_path, f"{figname}-map.{figext}")
    
    # values and required maptlotlib kwargs
    s = kwargs.get('std', 3)
    cmap = kwargs.get('cmap', 'rainbow')
    errormap = kwargs.get('error', None)

    ### handle required arguments
    value_string = f"{label} {units}"

    if ignore_mask_list:
        for i in ignore_mask_list:
            mask[mask == i] = 0
    
    image_mask = mask.astype(bool)

    masked_data = image[~image_mask]
    median = np.median(masked_data)
    standard_deviation = np.std(masked_data)

    plotmap = np.copy(image)
    plotmap[image_mask] = np.nan

    ###############################
    ######## historgram ###########
    ###############################

    #universal number of bins calculation
    bin_width = 3.5 * standard_deviation / (masked_data.size ** (1/3))
    nbins = (max(masked_data) - min(masked_data)) / bin_width
    plt.figure()
    plt.hist(masked_data,bins=int(nbins),color='k')
    plt.xlabel(label)
    plt.ylabel(r"$N_{\mathrm{bins}}$")
    plt.xlim(median - 7 * standard_deviation, median + 7 * standard_deviation)

    plt.savefig(histfigname, bbox_inches='tight')
    plt.close()
    print(f"{galname} {fig_keyword} histogram saved to {histfigname}")


    ###############################
    ########## STDMap #############
    ###############################
    vmin = median - s * standard_deviation
    vmax = median + s * standard_deviation

    plt.figure()
    im = plt.imshow(plotmap,cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',
               extent=[32.4, -32.6,-32.4, 32.6])
    plt.xlabel(r'$\Delta \alpha$ (arcsec)')
    plt.ylabel(r'$\Delta \delta$ (arcsec)')
    ax = plt.gca()
    ax.set_facecolor('lightgray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.01)

    cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
    cbar.set_label(value_string,fontsize=15,labelpad=-55)
    cax.xaxis.set_ticks_position('top')
    

    plt.savefig(stdfigname, bbox_inches='tight')
    plt.close()
    verbose_print(verbose, f"{galname} {fig_keyword} map plot saved to {stdfigname}")



    ## custom
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    minmax = kwargs.get('minmax', False)

    if vmin is not None or vmax is not None:
        stdcustfigname = os.path.join(fig_save_path, f"{figname}-custom-map.{figext}")
        plt.figure()
        if minmax:
            w = (image>vmax) & (image<vmin)
            plotmap[w] = np.nan
        im = plt.imshow(plotmap,cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',
               extent=[32.4, -32.6,-32.4, 32.6])
        plt.xlabel(r'$\Delta \alpha$ (arcsec)')
        plt.ylabel(r'$\Delta \delta$ (arcsec)')
        ax = plt.gca()
        ax.set_facecolor('lightgray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.01)

        cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
        cbar.set_label(value_string,fontsize=15,labelpad=-55)
        cax.xaxis.set_ticks_position('top')
        
        plt.savefig(stdcustfigname,bbox_inches='tight')
        plt.close()
        verbose_print(verbose, f"{galname} {fig_keyword} custom map plot saved to {stdcustfigname}")

    if errormap is not None:
        errormask = np.logical_or(image_mask, errormap<0)
        errormapfigname = os.path.join(fig_save_path, f"{figname}-map-error.{figext}")
        evmin = 0
        evmax = np.median(errormap[~errormask]) + 1 * np.std(errormap[~errormask])
        #error_string = r"$\sigma_{" + label[1:-1] + "}\ " + units[1:-1] + "$"
        error_string = fr"$\sigma_{{{label[1:-1]}}}\ {units[1:-1]}$"

        errormap[errormask] = np.nan
        plt.figure()
        im = plt.imshow(errormap,cmap=cmap,vmin=evmin,vmax=evmax,origin='lower',
               extent=[32.4, -32.6,-32.4, 32.6])
        plt.xlabel(r'$\Delta \alpha$ (arcsec)')
        plt.ylabel(r'$\Delta \delta$ (arcsec)')
        ax = plt.gca()
        ax.set_facecolor('lightgray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.01)

        cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
        cbar.set_label(error_string,fontsize=15,labelpad=-55)
        cax.xaxis.set_ticks_position('top')
        plt.savefig(errormapfigname,bbox_inches='tight')
        plt.close()
        verbose_print(verbose,f"{galname} {fig_keyword} uncertainty map plot saved to {errormapfigname}")



def make_bokeh_map(flux, model, ivar, wavelength, map, binid, output, map_keyword, palette="Turbo256", show=False):
    x, y = np.meshgrid(np.arange(map.shape[1]), np.arange(map.shape[0]))
    source = ColumnDataSource(data=dict(x=x.ravel(), y=y.ravel(), width=map.ravel(), bin=binid.ravel()))

    uncertainty = 1 / np.sqrt(ivar)


    std = np.std(map[np.isfinite(map)])
    low = np.median(map[np.isfinite(map)]) - std
    hi = np.median(map[np.isfinite(map)]) + std
    color_mapper = LinearColorMapper(palette=palette, low=low, high=hi)


    heatmap = bp.figure(title=map_keyword, tools="tap", 
                        x_axis_label='Spaxel', y_axis_label='Spaxel')
    heatmap.image(image=[map], x=0, y=0, dw=10, dh=10, color_mapper=color_mapper)
    heatmap.circle('x', 'y', size=10, source=source, alpha=0)


    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), label_standoff=12,
                         border_line_color=None, location=(0,0), title="")
    heatmap.add_layout(color_bar, 'right')


    flux_plot = bp.figure(title="Flux", x_axis_label='Wavelength', 
                          y_axis_label="Flux")
    uncertainty_plot = bp.figure(title="Uncertainty", 
                                 x_axis_label="Wavelength")
    model_plot = bp.figure(title="Stellar Model", 
                           x_axis_label="Wavelength")

    flux_source = ColumnDataSource(data=dict(x=[], y=[]))
    uncertainty_source = ColumnDataSource(data=dict(x=[], y=[]))
    model_source = ColumnDataSource(data=dict(x=[], y=[]))

    flux_plot.step('x', 'y', source=flux_source, color='black', mode='center')
    uncertainty_plot.line('x', 'y', source=uncertainty_source, color='dimgrey')
    model_plot.line('x', 'y', source=model_source, color='red', line_dash='dashed')

    bin_div = Div(text="<b>Bin:</b> ", width=200, height=30)

    callback = CustomJS(args=dict(source=source, flux_source=flux_source, uncertainty_source=uncertainty_source,
                                  model_source=model_source, bin_div=bin_div, flux=flux, uncertainty=uncertainty,
                                  model=model, wavelength=wavelength), code="""
                                  var indices = source.selected.indices;
                                  if (indices.length >0) {
                                    var index = indices[0]
                                    var x = source.data['x'][index];
                                    var y = source.data['y'][index];
                                    var bin_value = source.data['bin'][index]
                                    
                                    bin_div.text = "<b>Bin:</b> " + bin_value;

                                    var w_min = 5880;
                                    var w_max = 5910;
                                    var w_filtered = [];
                                    var f_filtered = [];
                                    var u_filtered = [];
                                    var u_filtered = [];
                                    var m_filtered = [];
                                    
                                    for (var i = 0; i < wavelength.length; i++) {
                                        if (wavelength[i][y][x] >= w_min && wavelength[i][y][x] <= w_max) {
                                            w_filtered.push(wavelength[i][y][x]);
                                            f_filtered.push(flux[i][y][x]);
                                            u_filtered.push(uncertainty[i][y][x]);
                                            m_filtered.push(model[i][y][x]);
                                        }
                                    }
                                    
                                    flux_source.data['x'] = w_filtered;
                                    flux_source.data['y'] = f_filtered;
                                    flux_source.change.emit();
                                    
                                    uncertainty_source.data['x'] = w_filtered;
                                    uncertainty_source.data['y'] = u_filtered;
                                    uncertainty_source.change.emit();
                                    
                                    model_source.data['x'] = w_filtered;
                                    model_source.data['y'] = m_filtered;
                                    model_source.change.emit();
                                }
                            """)
    heatmap.select(TapTool).callback = callback

    layout = column(heatmap, flux_plot, uncertainty_plot, model_plot)
    out_fname = os.path.join(output, f"{map_keyword}.html")
    output_file(out_fname)

    save(heatmap)
    print(f"BOKEH plot saved to {out_fname}")

#def plot_local_maps(mapsfil, verbose = True):
#    mapkeys = [field for field in hdul.keys() if 'MAP' in field.split('_')]
