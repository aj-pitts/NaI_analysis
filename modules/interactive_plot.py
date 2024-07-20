from bokeh.plotting import figure, save, output_file
from bokeh.io import curdoc
from bokeh.models import ColumnarDataSource, TapTool, CustomJS
from bokeh.layouts import column

import numpy as np

def make_bokeh_map(flux, model, ivar, wavelength, map, output, palette="Turbo256"):
    ly, lx = map.shape

    sigma = 1 / np.sqrt(ivar)
    depth = flux.shape[2]

    p = figure(title="Image Plot", x_range = (0, lx), y_range = (0, ly), tools='tap')
    p.image(image = [map], x=0, y=0, dw=lx, dh=ly, palette = palette)

    p_spectrum = figure(title="Flux", x_range=(5880, 5910), x_axis_label=r"Wavelength ($\mathrm{\AA}$)", 
                        y_axis_label=r"Flux ($\mathrm{10^{-17}\ erg\ cm^{-2}\ s\ spaxel}$)")
    
    spectrum_source = ColumnarDataSource(data={
        'wavelength':np.arange(depth), 
        'flux':np.zeros(depth),
        'uncertainty':np.zeros(depth), 
        'model':np.zeros(depth)
        })
    p_spectrum.line('Wavelength', 'flux', source=spectrum_source, legend_label='Flux', color="black", mode="center")
    p_spectrum.line('Wavelength', 'Uncertainty', source=spectrum_source, legend_label='Uncertainty', color='dimgray')
    p_spectrum.line('Wavelength', 'Model', source=spectrum_source, legend_label='Stellar Cont.', color="royalblue")

    callback = CustomJS(args=dict(source=spectrum_source, flux=flux, uncertainty=sigma,
                                  model=model), 
                        code="""
                    const indices = cb_data.index.indices;
                    if (indices.length > 0) {
                        const y = Math.floor(indices[0] / 100)};
                        const x = indices[0] % 100;
                        const spectrum_flux = flux.map((_, i) => flux[i][y][x];
                        const spectrum_uncertainty = uncertainty.map((_, i) => sigma[i][y][x]);
                        const spectrum_model = model.map((_, i) => model[i][y][x]);
                        source.data['flux'] = spectrum_flux;
                        source.data['Uncertainty'] = spectrum_uncertainty
                        source.data['model'] = spectrum_model;
                        source.change.emit();
                    }
                """)
    
    p.select(type=TapTool).callback = callback

    layout = column(p, p_spectrum)
    output_file(output)
    save(layout)
