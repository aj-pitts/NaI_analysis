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