import os
import shutil
import pandas as pd
import numpy as np
import astropy.units as u
import KeplerOrbit as ko
import pynbody as pb

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# Total number of initial condition files to generate
num_sims = 10

# The starting number for the first new simulation directory 
start_idx = 11

# Template directory to use for creating jobfiles
template_dir = 'template_expanse'

# Filename of the training data
training_file = 'stip_all.dat'

# Mass cut in the training set to split planetesimals and embryos.
# This is in units of log10 solar masses    
mcut_train = -7

# Number of epochs to use for training the GAN
train_epochs = 5000

# Number of simulations used to generate training data
n_train_sims = 5

# Filename postfix for the synthesizer model
model_file = '_all.pkl'

# Mass of the central star in the simulations (in solar masses)
mCentral = 0.08

# Bulk density of the planet-forming material (in g/cc)
pl_density = 3.0

def genICfile(pldf_emb, pldf_pl, icname):
    '''
    Given the training data set, uses the GAN to draw a collection of
    particles with the equivalent total mass and writes the result
    to an initial condition file for use with the N-body code genga

            Parameters:
                    pldf_emb (DataFrame): A pandas dataframe
                    containing the training data for the embryos
                    pldf_pl (DataFrame): A pandas dataframe
                    containing the training data for the planetesimals
                    icname (string): The name for the generated
                    initial condition file.
    '''

    synthesizer_emb = CTGANSynthesizer.load(filepath='emb' + model_file_post)
    synthesizer_pl = CTGANSynthesizer.load(filepath='pl' + model_file_post)

    # Enforce the constraint that the total mass of the synthetic disk
    # must match the total mass of the training data. Because the masses
    # are randomly drawn, the total particle counts will not match! What
    # we do here is to draw new particles in groups of npts_step until 
    # the total mass exceeds the target mass.
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

    pl_target_mass = np.sum(10**pldf_pl['log10mass'])/n_train_sims
    pl_gen = drawpts(synthesizer_pl, pl_target_mass, 50)

    emb_target_mass = np.sum(10**pldf_emb['log10mass'])/n_train_sims
    emb_gen = drawpts(synthesizer_emb, emb_target_mass, 1)

    df_gen = pd.concat([pl_gen, emb_gen])

    # Turn the generated data into a new initial condition file
    # First, the coordinates need to be converted back to cartesian

    # Orbital period needs to be converted into semimajor axis
    per_sim = (per*u.d).to(u.yr).value*(2*np.pi)
    a_vals = (mCentral*(per_sim/(2*np.pi))**2)**(1./3.)

    ecc_vals = 10**df_gen['log10e'].values
    inc_vals = 10**df_gen['log10inc'].values
    masses = 10**df_gen['log10mass'].values
    Omega_vals = np.random.rand(len(df_gen))*2*np.pi
    omega_vals = np.random.rand(len(df_gen))*2*np.pi
    M_vals = np.random.rand(len(df_gen))*2*np.pi

    # Storage for the cartesian coordinates
    positions = np.zeros((len(df_gen), 3))
    velocities = np.zeros_like(positions)

    for idx in range(len(df_gen)):
        p_x, p_y, p_z, v_x, v_y, v_z = ko.kep2cart(a_vals[idx], ecc_vals[idx], inc_vals[idx],\
                                               Omega_vals[idx], omega_vals[idx], M_vals[idx], masses[idx], mCentral)
        
        positions[idx] = p_x, p_y, p_z
        velocities[idx] = v_x, v_y, v_z
        
    density = (pl_density*u.g/u.cm**3).to(u.M_sun/u.AU**3).value
    radius = (3*masses/(4*np.pi*density))**(1./3.)

    # Write the cartesian coordinates, masses and particle radii to a genga IC file
    with open(icname, 'w') as f:
        for idx in range(len(df_gen)):
            line = str(positions[:,0][idx]) + ' ' + str(positions[:,1][idx]) + ' ' + str(positions[:,2][idx]) + ' ' + \
                   str(masses[idx]) + ' ' + \
                   str(velocities[:,0][idx]) + ' ' +  str(velocities[:,1][idx]) + ' ' +  str(velocities[:,2][idx]) + ' ' + \
                   str(radius[idx])
            f.write(line + '\n')

def main():
    # Load the training data and treat the specified columns as features
    columns = ['porb', 'log10mass', 'log10e', 'log10inc', 'sm_frac']
    pldf = pd.read_csv(training_file)
    pldf = pldf[columns]
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=pldf)

    # Model the posteriors of the planetsimals and embryos separately
    # this is done by making a mass cut in the training set
    pldf_emb = pldf[pldf['log10mass'] > mcut_train]
    pldf_pl = pldf[pldf['log10mass'] < mcut_train]

    # Generate the GAN model if it doesnt already exist
    if !os.file.isfile('emb' + model_file_post):
        synthesizer_emb = CTGANSynthesizer(metadata, epochs=train_epochs, verbose=True)
        synthesizer_emb.fit(pldf_emb)
        synthesizer_emb.save(filepath='emb' + model_file_post)

    if !os.file.isfile('pl' + model_file_post):
        synthesizer_pl = CTGANSynthesizer(metadata, epochs=train_epochs, verbose=True)
        synthesizer_pl.fit(pldf_pl)
        synthesizer_pl.save(filepath='pl' + model_file_post)

    # Generate {num_sims} new directories, each containing
    # a SLURM jobfile, an IC file, a genga parameter file
    # and a simulation output schedule
    for num in range(start_idx, start_idx + num_sims + 1):
        num_str = "{:02d}".format(num)
        dirpath = 'gen' + num_str
        print('Building ' + dirpath)

        # Duplicate the template directory
        shutil.copytree(template_dir, dirpath)

        # Generate the IC file
        icname = 'ic.dat'
        genICfile(pldf_emb, pldf_pl, icname)

        # Move the IC file into it
        os.rename(icname, dirpath + '/' + icname)

        # Update the jobfile name
        with open(dirpath + '/jobfile', 'r') as f:
          filedata = f.read()
        filedata = filedata.replace('{name}', dirpath)
        with open(dirpath + '/jobfile', 'w') as f:
          f.write(filedata)

if __name__ == "__main__":
    main()