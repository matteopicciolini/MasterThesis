import healpy as hp
import numpy as np
from halo import HaloCatalogManager

# Parameters for maps and catalogs
n_side = 4096
l_max = 3 * n_side - 1
n_pix = hp.nside2npix(n_side)

# Select halo catalogs with z < 0.5 and initialize the manager
snapshot_numbers = range(60, 63)
manager = HaloCatalogManager(snapshot_numbers)

# Set a mass threshold and apply a filter to halos
mass_threshold = 1e2
filtered_manager = manager.select_mass_above(mass_threshold)

halos = filtered_manager.get_all_halos()

# Spherical transform of the convergence map
#alm_kappa = hp.map2alm(halos.convergence_halos, lmax=l_max)
