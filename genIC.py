import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import astropy.units as u
import KeplerOrbit as ko
import pynbody as pb

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

mCentral = 0.08

def per2sma(per, perInDays=True):
    if perInDays:
        per_sim = (per*u.d).to(u.yr).value*(2*np.pi)
        return (mCentral*(per_sim/(2*np.pi))**2)**(1./3.)
    else:
        return (mCentral*(per/(2*np.pi))**2)**(1./3.)

def genICfile(icname):
    # Include smooth accretion fraction
    columns = ['porb', 'log10mass', 'log10e', 'log10inc', 'sm_frac']

    pldf = pd.read_csv('stip_all.dat')
    pldf = pldf[columns]
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=pldf)

    mcut = -7
    pldf_pl = pldf[pldf['log10mass'] < mcut]
    pldf_emb = pldf[pldf['log10mass'] > mcut]

    synthesizer_emb = CTGANSynthesizer.load(filepath='stip_emb_all_sm.pkl')
    synthesizer_pl = CTGANSynthesizer.load(filepath='stip_pl_all_sm.pkl')

    def drawpts(synthesizer, target_mass, npts_step):
        actual = 0
        npts = 1
        prev_max = 0.0

        while actual < target_mass:
            q = actual/target_mass*100.0
            if q > prev_max:
                prev_max = q
                print(f"{q:.2f}", npts)
            df_gen = synthesizer.sample(npts)
            actual = np.sum(10**df_gen['log10mass'])
            npts += npts_step

        return df_gen  

    pl_target_mass = np.sum(10**pldf_pl['log10mass'])/5
    pl_gen = drawpts(synthesizer_pl, pl_target_mass, 50)

    emb_target_mass = np.sum(10**pldf_emb['log10mass'])/5
    emb_gen = drawpts(synthesizer_emb, emb_target_mass, 1)

    df_gen = pd.concat([pl_gen, emb_gen])

    # Turn this into a pynbody snapshot and run it with genga

    a_vals = per2sma(df_gen['porb'].values)
    ecc_vals = 10**df_gen['log10e'].values
    inc_vals = 10**df_gen['log10inc'].values
    masses = 10**df_gen['log10mass'].values
    Omega_vals = np.random.rand(len(df_gen))*2*np.pi
    omega_vals = np.random.rand(len(df_gen))*2*np.pi
    M_vals = np.random.rand(len(df_gen))*2*np.pi

    positions = np.zeros((len(df_gen), 3))
    velocities = np.zeros_like(positions)

    for idx in range(len(df_gen)):
        p_x, p_y, p_z, v_x, v_y, v_z = ko.kep2cart(a_vals[idx], ecc_vals[idx], inc_vals[idx],\
                                               Omega_vals[idx], omega_vals[idx], M_vals[idx], masses[idx], mCentral)
        
        positions[idx] = p_x, p_y, p_z
        velocities[idx] = v_x, v_y, v_z
        
    density = (3*u.g/u.cm**3).to(u.M_sun/u.AU**3).value
    radius = (3*masses/(4*np.pi*density))**(1./3.)

    with open(icname, 'w') as f:
        for idx in range(len(df_gen)):
            line = str(positions[:,0][idx]) + ' ' + str(positions[:,1][idx]) + ' ' + str(positions[:,2][idx]) + ' ' + \
                   str(masses[idx]) + ' ' + \
                   str(velocities[:,0][idx]) + ' ' +  str(velocities[:,1][idx]) + ' ' +  str(velocities[:,2][idx]) + ' ' + \
                   str(radius[idx])
            f.write(line + '\n')

    print('Done')
