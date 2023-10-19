# synthetic_planet_ics

This program uses a collection of existing planet formation simulation data to generate a new snapshot which contains qualitatively similar but numerically distinct data. This is done by using a Generative Adversarial Network (GAN) to model the posterior distributions of the features in the dataset and then randomly drawing new points from the model. The model is powered by [CTGAN](https://github.com/sdv-dev/CTGAN).

This was used in the final chapter of my PhD thesis to generate a larger set of late-stage initial conditions for terrestrial planet formation simulations. Although the bulk of my thesis involved using high-resolution simulations to model this process starting from the smallest gravitationally bound objects, planetesimals, I found that it was too computationally expensive to run a statistically robust sample of planetary systems all the way to completion. In particular, the early stages of growth are the most calculation-intensive and the simulations become cheaper as they evolve because the particle count diminishes as objects conglomerate. At the same time, broader regions of the planet-forming disk come into communication with each other and the system becomes chaotic. To fully understand the outcome of this process, one woud need to run a large number of simulations through the chaotic phase.

To circumvent this issue, I train the GAN on some of the intermediate simulation snapshots, and use the model to generate a much larger set of initial conditions that begin partway through the planet formation process. Because the ICs are all numerically distinct, I can run them to gain a broader picture of the possible outcomes of the dynamical chaos.