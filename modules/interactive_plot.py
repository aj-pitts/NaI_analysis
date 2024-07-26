import bokeh.plotting as bp
from bokeh.models import ColumnDataSource, TapTool, CustomJS, ColorBar, LinearColorMapper
from bokeh.layouts import column
from bokeh.transform import linear_cmap
from bokeh.io import output_file, show
from bokeh.models import BasicTicker, ColorBar

import numpy as np

def make_bokeh_map(flux, model, ivar, wavelength, map, binid, output, map_keyword, palette="Turbo256", show=False):
    x, y = np.meshgrid(np.arange(map.shape[1]), np.arange(map.shape[0]))
    source = ColumnDataSource(data=dict(x=x.ravel(), y=y.ravel(), width=map.ravel()))

    uncertainty = 1 / np.sqrt(ivar)


    std = np.std(map[np.isfinite(map)])
    low = np.median(map[np.isfinite[map]]) - std
    hi = np.median(map[np.isfinite[map]]) + std
    color_mapper = LinearColorMapper(palette=palette, low=low, high=hi)


    heatmap = bp.figure(title=map_keyword, tools="tap", plot_width=400, plot_height=400, 
                        x_axis_label='Spaxel', y_axis_label='Spaxel')
    heatmap.image(image=[map], x=0, y=0, dw=10, dh=10, palette=palette, color_mapper=color_mapper)
    heatmap.circle('x', 'y', size=10, source=source, alpha=0)


    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), label_standoff=12,
                         border_line_color=None, location=(0,0), title="")
    heatmap.add_layout(color_bar, 'right')


    flux_plot = bp.figure(title="Flux", plot_width=400, plot_height=200, x_axis_label='Wavelength', 
                          y_axis_label="Flux")
    uncertainty_plot = bp.figure(title="Uncertainty", plot_width=400, plot_height=200, 
                                 x_axis_label="Wavelength")
    model_plot = bp.figure(title="Stellar Model", plot_width=400, plot_height=200, 
                           x_axis_label="Wavelength")

    flux_source = ColumnDataSource(data=dict(x=[], y=[]))
    uncertainty_source = ColumnDataSource(data=dict(x=[], y=[]))
    model_source = ColumnDataSource(data=dict(x=[], y=[]))

    flux_plot.step('x', 'y', source=flux_source, color='black', model='center')
    uncertainty_plot.line('x', 'y', source=uncertainty_source, color='dimgrey')
    model_plot.line('x', 'y', source=model_source, color='red', line_dash='dashed')

    callback = CustomJS(args=dict(source=source, flux_source=flux_source, uncertainty_source=uncertainty_source,
                                  model_source=model_source, flux=flux, uncertainty=uncertainty,
                                  model=model, wavelength=wavelength), code="""
                                  var indices = source.selected.indices;
                                  if (indices.length >0) {
                                    var index = indices[0]
                                    var x = source.data['x'][index];
                                    var y = source.data['y'][index];
                                    
                                    var w_min = 5880;
                                    var w_max = 5910;
                                    var w_filtered = [];
                                    var f_filtered = [];
                                    var u_filtered = [];
                                    var u_filtered = [];
                                    var m_filtered = [];
                                    
                                    for (var i = 0; i < wavelength.length; i++) {
                                        if (wavelength[i] >= w_min && wavelength[i] <= w_max) {
                                            w_filtered.push(wavelength[i]);
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
