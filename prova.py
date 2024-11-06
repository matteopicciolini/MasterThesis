import numpy as np
from astropy.cosmology import LambdaCDM
import astropy.units as u

# Parametri cosmologici dalla tua simulazione
OmegaCDM = 0.27  # Densità di materia oscura
OmegaBr = 0.05   # Densità di materia barionica
OmegaNeu = 0.00  # Densità di neutrini
OmegaLambda = 0.68  # Densità di energia oscura
h_Hubble = 0.67  # Hubble normalizzato

# Creazione dell'oggetto LambdaCDM
cosmo = LambdaCDM(H0=h_Hubble * 100, Om0=OmegaCDM + OmegaBr, Ode0=OmegaLambda)

# Redshift della CMB
z_cmb = 1100  # Redshift della CMB
z_1 = 49.4806
# Calcolo della distanza comovente a redshift z_cmb
D_comov = cosmo.comoving_distance(z_cmb)
D_comov2 = cosmo.comoving_distance(z_1)
print((D_comov-D_comov2)/D_comov)