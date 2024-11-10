import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import defaults
from util import verbose_logger

import bokeh.plotting as bp
from bokeh.models import ColumnDataSource, TapTool, CustomJS, ColorBar, LinearColorMapper, Div
from bokeh.layouts import column
from bokeh.transform import linear_cmap
from bokeh.io import output_file, show, save
from bokeh.models import BasicTicker, ColorBar

def map_plotter(image: np.ndarray, mask: np.ndarray, fig_keyword: str, label: str, units: str, galname: str, 
             bin_method: str, verbose = True, **kwargs):
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
        | figpath     | str        | The absolute filepath to the figures directory.              |
        |             |            | figures directory.                                           |
        | figname     | str        | The leading name of each figure file to be written.          |
        | figext      | str        | The filetype extension of the figure file.                   |
        | error       | np.ndarray | Map of propagated uncertainties corresponding to `image`.    |
        
    """
    ### init verbose logger
    logger = verbose_logger(verbose)

    plt.style.use(defaults.matplotlib_rc())
    default_figpath = defaults.get_default_path('local_outputs/figures')

    ### unpack and handle defaults for all kwargs
    # names and paths
    figpath = kwargs.get('figpath', default_figpath)
    figext = kwargs.get('figext', 'pdf')
    figname = kwargs.get('figname', f'{galname}-{bin_method}-{fig_keyword}')

    histfigname = os.path.join(figpath, f"{figname}-histogram.{figext}")
    stdfigname = os.path.join(figpath, f"{figname}-map.{figext}")
    
    # values and required maptlotlib kwargs
    s = kwargs.get('std', 3)
    cmap = kwargs.get('cmap', 'rainbow')
    dpi = kwargs.get('dpi', 200)
    errormap = kwargs.get('error', None)

    ### handle required arguments
    mask = mask.astype(bool)
    masked_data = image[~mask]

    median = np.median(image[~mask])
    standard_deviation = np.std(image[~mask])

    value_string = f"{label} {units}"


    #### make the plots
    ## histogram

    #flatten data
    flatdata = np.copy(masked_data).flatten()

    #universal number of bins calculation
    bin_width = 3.5 * np.std(flatdata) / (flatdata.size ** (1/3))
    nbins = (max(flatdata) - min(flatdata)) / bin_width

    plt.hist(flatdata,bins=int(nbins),color='k')
    plt.xlabel(label)
    plt.ylabel(r"$N_{\mathrm{bins}}$")
    plt.xlim(median - 7 * standard_deviation, median + 7 * standard_deviation)

    plt.savefig(histfigname, bbox_inches='tight')
    plt.close()
    logger.info(f"{galname} {fig_keyword} histogram saved to {histfigname}")


    ## STD map
    vmin = median - s * standard_deviation
    vmax = median + s * standard_deviation

    plotmap = np.copy(image)
    plotmap[mask] = np.nan

    im = plt.imshow(plotmap,cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',
               extent=[32.4, -32.6,-32.4, 32.6])
    plt.xlabel(r'$\alpha$ (arcsec)')
    plt.ylabel(r'$\delta$ (arcsec)')
    ax = plt.gca()
    ax.set_facecolor('lightgray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.01)

    cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
    cbar.set_label(value_string,fontsize=15,labelpad=-45)
    cax.xaxis.set_ticks_position('top')
    

    plt.savefig(stdfigname, bbox_inches='tight',dpi=dpi)
    plt.close()
    logger.info(f"{galname} {fig_keyword} map plot saved to {stdfigname}")



    ## custom
    vmin = kwargs.get('vmin',None)
    vmax = kwargs.get('vmax',None)

    if vmin or vmax:
        stdcustfigname = os.path.join(figpath, f"{figname}-custom-map.{figext}")
        im = plt.imshow(plotmap,cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',
               extent=[32.4, -32.6,-32.4, 32.6])
        plt.xlabel(r'$\alpha$ (arcsec)')
        plt.ylabel(r'$\delta$ (arcsec)')
        ax = plt.gca()
        ax.set_facecolor('lightgray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.01)

        cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
        cbar.set_label(value_string,fontsize=15,labelpad=-45)
        cax.xaxis.set_ticks_position('top')
        
        plt.savefig(stdcustfigname,bbox_inches='tight',dpi=dpi)
        plt.close()
        logger.info(f"{galname} {fig_keyword} custom map plot saved to {stdcustfigname}")

    if errormap:
        errormapfigname = os.path.join(figpath, f"{figname}-map-error.{figext}")
        vmin = np.median(errormap[~mask]) - s * np.std(errormap[~mask])
        vmax = np.median(errormap[~mask]) + s * np.std(errormap[~mask])
        error_string = rf"$\sigma_{{{label}}}$ {units}"

        errormap[mask] = np.nan
        im = plt.imshow(errormap,cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',
               extent=[32.4, -32.6,-32.4, 32.6])
        plt.xlabel(r'$\alpha$ (arcsec)')
        plt.ylabel(r'$\delta$ (arcsec)')
        ax = plt.gca()
        ax.set_facecolor('lightgray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.01)

        cbar = plt.colorbar(im,cax=cax,orientation = 'horizontal')
        cbar.set_label(error_string,fontsize=15,labelpad=-45)
        cax.xaxis.set_ticks_position('top')
        
        plt.savefig(errormapfigname,bbox_inches='tight',dpi=dpi)
        plt.close()
        logger.info(f"{galname} {fig_keyword} uncertainty map plot saved to {stdcustfigname}")
        




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

def plot_local_maps(mapsfil, verbose = True):
    logger = verbose_logger(verbose=verbose)
    hdul = fits.open(mapsfil)
    mapkeys = [field for field in hdul.keys() if 'MAP' in field.split('_')]
