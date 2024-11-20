from halo import *
import random
import matplotlib.pyplot as plt

# Parameters for maps and catalogs
n_side = 512
l_max = 3 * n_side - 1
n_pix = hp.nside2npix(n_side)

# Select halo catalogs with z < 1 and initialize the manager
snapshot_numbers = range(40, 63)
manager = HaloCatalogManager(snapshot_numbers)

# Set a mass threshold and apply a filter to halos
mass_threshold = 1e3
filtered_manager = manager.select_mass_above(mass_threshold)

# get all halos
halos = filtered_manager.get_all_halos(n_side)
print(f"Total halos are {halos.n_halos}")
print(f"Pixels are {n_pix}")

# build map of convergence form uniform distribution GW event
convergence_map = np.zeros(n_pix)
j=1
for i in range(int(1e6)):
    rnd_index = random.randint(0, halos.n_halos-1)
    rnd_halo = halos.get_halo(rnd_index);
    if convergence_map[rnd_halo.pixel]==0:
        convergence_map[rnd_halo.pixel] = rnd_halo.convergence
    else:
        j+=1

print(f"Pixel already filled are {j}")
hp.mollview(convergence_map, title="Convergence from 1e6 GW signals within z=1", unit="Convergence", norm="hist")
hp.graticule()
plt.show()

halos_map = np.zeros(n_pix)
for i in range(halos.n_halos):
    single_halo = halos.get_halo(i);
    halos_map[single_halo.pixel] += 1

#------------------ HALOS ----------------------

mean_halos_count = np.mean(halos_map)
halos_contrast = (halos_map - mean_halos_count) / mean_halos_count
cross_correlation_kappa = hp.anafast(halos_contrast, convergence_map)


hp.mollview(halos_map, title=r'Halos Map (Mass > $10^{13} M_{\odot}$/h)', unit='Contrast of Halos')
hp.graticule()

# ---------------- CROSS ----------------------

l = np.arange(len(cross_correlation_kappa))
plt.figure(figsize=(15, 5))

# Linear scale
plt.subplot(1, 2, 1)
plt.plot(l, cross_correlation_kappa * l * (l + 1)/(2 * np.pi), label='Cross-correlation', color='orange')
plt.xlabel('Multipole moment (l)')
plt.yscale('log')
plt.ylabel(r'$C_\ell^{N\kappa} \cdot \ell(\ell+1)/2\pi$')
plt.title('Cross-correlation (Linear Scale)', fontsize=15)
plt.legend()
plt.grid()

# Log scale
plt.subplot(1, 2, 2)
plt.plot(l, cross_correlation_kappa * l * (l + 1)/(2 * np.pi), label='Cross-correlation', color='orange')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Multipole moment (l)')
plt.ylabel(r'$C_\ell^{N\kappa} \cdot \ell(\ell+1)/2\pi$')
plt.title('Cross-correlation (Log Scale)', fontsize=15)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


#################################################################


convergence_map = np.zeros(n_pix)
j=1
for i in range(halos.n_halos):
    rnd_halo = halos.get_halo(i);
    if convergence_map[rnd_halo.pixel]==0:
        convergence_map[rnd_halo.pixel] = rnd_halo.convergence
    else:
        j+=1

print(f"Pixel already filled are {j}")
hp.mollview(convergence_map, title="Convergence from all halos GW signals within z=1", unit="Convergence", norm="hist")
hp.graticule()
plt.show()


#-------------------- cross ----------------------
cross_correlation_kappa = hp.anafast(halos_contrast, convergence_map)



l = np.arange(len(cross_correlation_kappa))
plt.figure(figsize=(15, 5))

# Linear scale
plt.subplot(1, 2, 1)
plt.plot(l, cross_correlation_kappa * l * (l + 1)/(2 * np.pi), label='Cross-correlation', color='orange')
plt.xlabel('Multipole moment (l)')
plt.yscale('log')
plt.ylabel(r'$C_\ell^{N\kappa} \cdot \ell(\ell+1)/2\pi$')
plt.title('Cross-correlation (Linear Scale)', fontsize=15)
plt.legend()
plt.grid()

# Log scale
plt.subplot(1, 2, 2)
plt.plot(l, cross_correlation_kappa * l * (l + 1)/(2 * np.pi), label='Cross-correlation', color='orange')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Multipole moment (l)')
plt.ylabel(r'$C_\ell^{N\kappa} \cdot \ell(\ell+1)/2\pi$')
plt.title('Cross-correlation (Log Scale)', fontsize=15)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
