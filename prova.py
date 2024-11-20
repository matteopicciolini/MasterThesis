import numpy as np
from astropy.cosmology import LambdaCDM
import astropy.units as u

# Parametri cosmologici dalla tua simulazione
OmegaCDM = 0.27  # Densità di materia oscura
OmegaBr = 0.05   # Densità di materia barionica
OmegaNeu = 0.00  # Densità di neutrini
OmegaLambda = 0.68  # Densità di energia oscura
h_Hubble = 0.67  # Hubble

# Creazione dell'oggetto LambdaCDM
cosmo = LambdaCDM(H0=h_Hubble * 100, Om0=OmegaCDM + OmegaBr, Ode0=OmegaLambda)

# Redshift della CMB
z_cmb = 0.5  # Redshift della CMB
z_1 = 0.018150
z_2 = 0.055450

D_comov_1 = cosmo.comoving_distance(z_1)
print(D_comov_1 * 1e3 * h_Hubble)

D_comov_2= cosmo.comoving_distance(z_2)
print(D_comov_2 * 1e3 * h_Hubble)