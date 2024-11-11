import numpy as np
import healpy as hp
import os
from astropy.cosmology import LambdaCDM


class HaloCatalog:
    """
    Class representing a catalog of dark matter halos.

    Attributes:
    -----------
    dtype_halo : numpy.dtype
        Defines the binary data structure for halos.
    snapshot_number : int
        Number indicating the snapshot.
    redshift : float
        The redshift corresponding to the snapshot.
    halos : numpy.ndarray
        Array containing halo data.
    n_halos : int
        The number of halos in the catalog.
    """
    # Define the binary data type for halos
    dtype_halo = np.dtype([
        ('theta', 'd'), ('phi', 'd'), ('dr', 'd'),
        ('Mass', 'd'), ('pos_x', 'd'), ('pos_y', 'd'),
        ('pos_z', 'd'), ('vel_x', 'd'), ('vel_y', 'd'),
        ('vel_z', 'd')
    ])

    def __init__(self, snapshot_number, halos=None):
        """
        Initializes a new HaloCatalog.

        Parameters:
        -----------
        snapshot_number : int
            The snapshot number corresponding to the current catalog.
        halos : numpy.ndarray, optional
            Array containing halo data. If None, data is loaded via the load() method.
        """
        self.snapshot_number = snapshot_number
        self.redshift = self.find_redshift()
        if halos is not None:
            self.halos = halos
        else:
            self.halos = self.load()
        self.n_halos = self.count()

    def read_halos(self, file_name):
        """Reads the binary data from a single file and returns the halo data."""
        return np.fromfile(file_name, dtype=self.dtype_halo)

    def load(self):
        """
        Loads halo data by combining binary files for the snapshot.

        Returns:
        --------
        numpy.ndarray
            Array containing the combined halo data from all binary files.
        """
        # Define the list of binary files to load for the given snapshot
        files = [f"AllSkyMock_snap_{self.snapshot_number:03d}_{i}.bin0" for i in range(4)]

        all_halos = []

        for file in files:
            if os.path.exists(file):
                print(f"Reading {file}")
                halos = self.read_halos(file)
                all_halos.append(halos)
            else:
                print(f"File not found: {file}")

        # Concatenate all halo data from the loaded files
        return np.concatenate(all_halos)

    def count(self):
        """Returns the number of halos in self.halos."""
        return len(self.halos)

    def find_redshift(self):
        """
        Finds the redshift corresponding to the snapshot number.

        Returns:
        --------
        float
            The redshift corresponding to the snapshot number, or None if not found.
        """
        simulations_data = np.loadtxt('new_cosmoint_results.txt', usecols=(0, 1))
        simulation_snapshots = simulations_data[:, 0]
        simulation_redshifts = simulations_data[:, 1]
        snapshot_to_redshift = dict(zip(simulation_snapshots, simulation_redshifts))
        return snapshot_to_redshift.get(self.snapshot_number)

    @property
    def theta(self):
        """Returns the theta values for all halos."""
        return self.halos['theta']

    @property
    def phi(self):
        """Returns the phi values for all halos."""
        return self.halos['phi']

    @property
    def mass(self):
        """Returns the mass values of halos."""
        return self.halos['Mass']

    def select_mass_above(self, mass_limit: float) -> 'HaloCatalog':
        """
        Filters halos based on a mass limit and returns a new instance of HaloCatalog.

        Parameters:
        -----------
        mass_limit : float
            The mass limit to filter halos.

        Returns:
        --------
        HaloCatalog
            A new instance of HaloCatalog containing only halos above the mass limit.
        """
        mask = self.halos['Mass'] > mass_limit
        filtered_halos = self.halos[mask]
        return HaloCatalog(self.snapshot_number, filtered_halos)

    def calculate_velocity(self) -> np.ndarray:
        """
        Calculates the velocity vector for each halo.

        Returns:
        --------
        numpy.ndarray
            Array containing the magnitudes of the velocity vectors for each halo.
        """
        vel_x = self.halos['vel_x']
        vel_y = self.halos['vel_y']
        vel_z = self.halos['vel_z']
        velocities = np.vstack((vel_x, vel_y, vel_z)).T  # Stack velocity components into a 2D array
        return np.linalg.norm(velocities, axis=1) * (1 + self.redshift) # Compute the magnitude of velocity for each halo


class HaloCatalogManager:
    """
    Manages multiple halo catalogs from different snapshots.

    Attributes:
    -----------
    snapshot_numbers : list
        List of snapshot numbers corresponding to the catalogs.
    catalogs : list
        List of HaloCatalog objects. If None, HaloCatalogs will be created for each snapshot number.
    """

    def __init__(self, snapshot_numbers, catalogs=None):
        """
        Initializes a new HaloCatalogManager.

        Parameters:
        -----------
        snapshot_numbers : list
            List of snapshot numbers corresponding to the catalogs.
        catalogs : list, optional
            List of HaloCatalog objects. If None, HaloCatalogs will be created for each snapshot number.
        """
        if catalogs is not None:
            self.snapshot_numbers = snapshot_numbers
            self.catalogs = catalogs
        else:
            # Create HaloCatalog objects for each snapshot number if catalogs are not provided
            self.snapshot_numbers = snapshot_numbers
            self.catalogs = [HaloCatalog(snapshot_number) for snapshot_number in snapshot_numbers]

    def get_all_halos_combined(self):
        """
        Combines halos from all catalogs into a single array.

        Returns:
        --------
        numpy.ndarray
            Array containing all halos from all catalogs.
        """
        all_halos = np.concatenate([catalog.halos for catalog in self.catalogs])
        return all_halos

    def get_all_halos(self, n_side: int = 4096) -> 'Halos':
        """
        Combines all halos and maps them to Healpix pixels.

        Parameters:
        -----------
        n_side : int, optional
            The nside parameter for Healpix (default is 4096).

        Returns:
        --------
        AllHalosFromCatalogs
            An instance of AllHalosFromCatalogs containing all halos and their Healpix mapping.
        """
        theta = np.concatenate(self.theta)
        phi = np.concatenate(self.phi)
        snapshots = np.concatenate(self.snapshot)
        redshifts = np.concatenate(self.redshift)
        return Halos(theta, phi, snapshots, redshifts, self.snapshot_numbers, n_side)

    def select_mass_above(self, mass_limit: float) -> 'HaloCatalogManager':
        """
        Filters halos in all catalogs based on a mass limit and returns a new HaloCatalogManager.

        Parameters:
        -----------
        mass_limit : float
            The mass limit to filter halos.

        Returns:
        --------
        HaloCatalogManager
            A new instance of HaloCatalogManager containing only halos above the mass limit.
        """
        filtered_catalogs = [catalog.select_mass_above(mass_limit) for catalog in self.catalogs]
        return HaloCatalogManager(self.snapshot_numbers, catalogs=filtered_catalogs)

    def calculate_velocity(self) -> list[np.ndarray]:
        """
        Calculates velocity for each catalog and returns a list of velocity arrays.

        Returns:
        --------
        list of numpy.ndarray
            A list containing arrays of velocity magnitudes for each catalog.
        """
        return [catalog.calculate_velocity() for catalog in self.catalogs]

    @property
    def theta(self) -> list[np.ndarray]:
        """Returns the theta values for all catalogs."""
        return [catalog.theta for catalog in self.catalogs]

    @property
    def mass(self) -> list[np.ndarray]:
        """Returns the mass values for all catalogs."""
        return [catalog.mass for catalog in self.catalogs]

    @property
    def phi(self) -> list[np.ndarray]:
        """Returns the phi values for all catalogs."""
        return [catalog.phi for catalog in self.catalogs]

    @property
    def snapshot(self) -> list[np.ndarray]:
        """Returns an array of snapshot numbers for all halos in each catalog."""
        return [np.full(len(catalog.halos), catalog.snapshot_number) for catalog in self.catalogs]

    @property
    def redshift(self) -> list[np.ndarray]:
        """Returns an array of redshifts for all halos in each catalog."""
        return [np.full(len(catalog.halos), catalog.redshift) for catalog in self.catalogs]

def lens_weight(z_l, z_s=1100):
    # Cosmological parameters
    OmegaCDM = 0.27  # Dark matter density
    OmegaBr = 0.05  # Baryonic matter density
    OmegaNeu = 0.00  # Neutrino density
    OmegaLambda = 0.68  # Dark energy density
    h_Hubble = 0.67  # Normalized Hubble parameter

    # Initialize the cosmological model
    cosmo = LambdaCDM(H0=h_Hubble * 100, Om0=OmegaCDM + OmegaBr, Ode0=OmegaLambda)

    # Calculate the comoving distance for `z_l`
    d_comov_l = np.array(cosmo.comoving_distance(z_l))

    # If `z_s` is a scalar, calculate directly
    if np.isscalar(z_s):
        if z_s == 0.:
            return 0.  # Return 0 if `z_s` is zero
        d_comov_s = np.array(cosmo.comoving_distance(z_s))
        return (d_comov_s - d_comov_l) / d_comov_s

    # If `z_s` is an array, calculate element by element
    d_comov_s = np.array(cosmo.comoving_distance(z_s))

    # Calculate the lens weight with handling for elements in `z_s` equal to 0
    np.seterr(invalid='ignore')
    lens_weights = np.where(np.isclose(d_comov_s, 0.), 0., (d_comov_s - d_comov_l) / d_comov_s)

    return lens_weights


class Halos:
    """
    Represents all halos from multiple catalogs and maps them to Healpix pixels.

    Attributes:
    -----------
    theta : numpy.ndarray
        Array of theta values for all halos.
    phi : numpy.ndarray
        Array of phi values for all halos.
    pixels : numpy.ndarray
        Array of Healpix pixel indices for all halos.
    snapshots : numpy.ndarray
        Array of snapshot numbers for all halos.
    redshifts : numpy.ndarray
        Array of redshifts for all halos.
    snapshot_numbers : list
        List of snapshot numbers corresponding to the catalogs.
    n_side : int
        The nside parameter for Healpix.
    path : str
        The path where the convergence maps are located.
    convergence_halos : numpy.ndarray
        Array containing the halo convergence data.
    """

    def __init__(self, theta, phi, snapshots, redshifts, snapshot_numbers, n_side=4096, path=""):
        """
        Initializes a new AllHalosFromCatalogs.

        Parameters:
        -----------
        theta : numpy.ndarray
            Array of theta values for all halos.
        phi : numpy.ndarray
            Array of phi values for all halos.
        snapshots : numpy.ndarray
            Array of snapshot numbers for all halos.
        redshifts : numpy.ndarray
            Array of redshifts for all halos.
        snapshot_numbers : list
            List of snapshot numbers corresponding to the catalogs.
        n_side : int, optional
            The nside parameter for Healpix (default is 4096).
        path : str, optional
            The path where the convergence maps are located (default is an empty string).
        """
        # Single values
        self.snapshot_numbers = snapshot_numbers  # Without repetition
        self.n_side = n_side
        self.path = path


        # Halos data
        self.theta = theta
        self.phi = phi
        self.pixels = hp.ang2pix(n_side, theta, phi)  # Convert angular coordinates to Healpix pixel indices
        self.snapshots = snapshots
        self.redshifts = redshifts
        self.n_halos = len(self.theta) # Single value
        self.convergence = self.update_convergence()






    def update_convergence(self):
        """
        Updates the halo convergence by adding halo data from each snapshot.

        Returns:
        --------
        numpy.ndarray
            Array containing the updated halo convergence data.
        """


        data = np.loadtxt(f'{self.path}new_cosmoint_results.txt', skiprows=1)
        cmb_weights = data[:, -1]
        z_ls = data[:, 1]

        convergence_halos = np.zeros(self.n_halos)
        for snapshot in self.snapshot_numbers[::-1]:
            cmb_wight = cmb_weights[np.where(data[:, 0] == snapshot)[0][0]]
            z_l = z_ls[np.where(data[:, 0] == snapshot)[0][0]]

            map_file = f'{self.path}KappaMap_snap_{snapshot:03d}.DM.seed_100672.fits'
            print(f"Reading {map_file}")

            # Read the convergence map for the current snapshot
            convergence_map = (1./cmb_wight) * hp.read_map(map_file, dtype=np.float32)

            mask = self.snapshots < snapshot
            l_weights = lens_weight(z_l, self.redshifts[mask])
            convergence_halos[mask] += l_weights * convergence_map[self.pixels[mask]]

            # Remove temporary variables to free memory
            del convergence_map
            del mask
            del l_weights


        return convergence_halos
