from __future__ import print_function

import math
import numpy as np
import os
import fnmatch
import scipy.optimize as op
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.io as pio
import pdb

outfil = '../../2019nov/Figs/fig_sfrM_MAD.pdf'

## Read in SDSS info
mpajhufil = '/Users/rubin/Research/data/MaNGA/Catalogs/galSpecExtra-dr8.fits'
mpajhu = fits.open(mpajhufil)[1].data
sdssind = (mpajhu['LGM_TOT_P50'] > 5.0) & (mpajhu['SFR_TOT_P50'] > -10.0)
## Use mpajhu['SFR_TOT_P50'] -- log SFR
##     mpajhu['LGM_TOT_P50'] -- log Mstar

## Read in MaNGA info
NaIfil = '/Users/rubin/Research/MaNGA/NaImcmc/Figures/py/NaImcmc_velinfo_2019feb04_FluxFit_SN30_vflowmin60.fits'
NaIfits = fits.getdata(NaIfil)
area_min = 0.5
mangaind = (NaIfits['AREAFLG'] > area_min)
MaNGA_logMstar = np.log10((10.0**NaIfits['STELLAR_MASS']) / (0.7**2))
## NaIfits['LOGSFR'], NaIfits['STELLAR_MASS'], NaIfits['AREAFLG'] > area_min

## Read in MAD
madfil = 'MAD_sample.dat'
mad = ascii.read(madfil)
madlogMstar = mad['col6']
madlogSFR = np.log10(mad['col7'])
#pdb.set_trace()
selmph = np.where((mad['col1']=='NGC1512') | (mad['col1']=='NGC2835'))

## Read in PHANGS
phangsfil = '../../2019nov/Figs/PHANGS_sample.dat'
phangs = ascii.read(phangsfil)

phangslogMstar = phangs['Mstar']
phangslogSFR = phangs['logSFR']


trace1 = go.Histogram2dContour(
    x=mpajhu['LGM_TOT_P50'][sdssind], y=mpajhu['SFR_TOT_P50'][sdssind], name='SDSS (Brinchmann et al. 2004)', ncontours=30,
    colorscale='Greys', reversescale=True, showscale=False
    )
trace2 = go.Scatter(
    x=MaNGA_logMstar[mangaind], y=NaIfits['LOGSFR'][mangaind], name='MaNGA High S/N', mode='markers',
    marker=dict(size=4, color='orange'))
trace3 = go.Scatter(
    x=madlogMstar, y=madlogSFR, mode='markers', name='MUSE Atlas of Disks',
    marker=dict(size=8, color='red', symbol='circle'))
trace4 = go.Scatter(
    x=madlogMstar[selmph], y=madlogSFR[selmph], mode='markers', showlegend=False,
    marker=dict(size=11, color='red', symbol='circle',line=dict(color='blue', width=2)))
trace5 = go.Scatter(
    x=phangslogMstar[:-2], y=phangslogSFR[:-2], mode='markers', name='PHANGS',
    marker=dict(size=11, color='white', symbol='circle', line=dict(color='blue', width=2)))

data = [trace1, trace3, trace5]

layout = go.Layout(showlegend=True, autosize=False, width=600, height=550,
                    legend=dict(x=0,y=1,font=dict(size=18, color='black', family='Times New Roman')),
                    annotations=[dict(x=0.5, y=-0.15, showarrow=False, text='$\log M_{*}/M_{\odot}$', xref='paper', yref='paper',
                                          font=dict(size=40, color='black')),
                                     dict(x=-0.18, y=0.5, showarrow=False, text='$\log \mathrm{SFR}/[M_{\odot}\mathrm{yr}^{-1}]$)',
                                              xref='paper', yref='paper', textangle=-90, 
                                              font=dict(size=40, color='black'))],
                   xaxis=dict(range=[8.4,12], domain=[0,1.0],
                              showgrid=True, zeroline=True, showline=True, linewidth=2, mirror='ticks', 
                              #titlefont=dict(color='black', size=20, family='Times New Roman'),
                              tickfont=dict(color='black', size=20, family='Times New Roman')),
                   yaxis=dict(range=[-2.1,1.5], domain=[0,1.0],
                              showgrid=True, zeroline=True, showline=True, linewidth=2, mirror='ticks',
                              #titlefont=dict(color='black', size=20, family='Times New Roman'),
                              tickfont=dict(color='black', size=20, family='Times New Roman')))


fig = go.Figure(data=data, layout=layout)
fig.add_trace(trace4)

pio.write_image(fig, outfil)                              
#pdb.set_trace()
