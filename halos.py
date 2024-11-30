import os
import numpy as np
from multiprocessing import Pool
from astropy.cosmology import LambdaCDM, z_at_value
import healpy as hp
import astropy.units as u
import astropy.constants as astro
from tqdm import tqdm

def compute_z(args):
    """
    Compute the redshift corresponding to a given comoving distance.

    Parameters
    ----------
    args : tuple
        A tuple containing the following elements:
        - distance (float): Comoving distance in kpc.
        - cosmo (astropy.cosmology.LambdaCDM): Cosmology model instance.
        - z_min (float): Minimum redshift for search range.
        - z_max (float): Maximum redshift for search range.
        - h_hubble (float): Hubble parameter normalization factor.

    Returns
    -------
    float
        Computed redshift value.
    """
    distance, cosmo, z_min, z_max, h_hubble = args
    # Calculate the redshift as a Quantity, then extract its numerical value
    z = z_at_value(cosmo.comoving_distance, distance * u.kpc / h_hubble, zmin=z_min - 1e-4, zmax=z_max + 1e-4)
    return z.value


class Cosmo:
    """
    A class to handle cosmological computations, including redshift calculations,
    lensing weights, and conformal Hubble parameter.

    Attributes
    ----------
    cosmo : astropy.cosmology.LambdaCDM
        The cosmological model based on ΛCDM.
    h_Hubble : float
        Hubble parameter normalization factor (dimensionless).
    """

    def __init__(self):
        """
        Initialize the cosmological model with standard ΛCDM parameters.
        """
        OmegaCDM = 0.27  # Cold dark matter density
        OmegaBr = 0.05   # Baryonic matter density
        OmegaNeu = 0.00  # Neutrino density (not used but could be added later)
        OmegaLambda = 0.68  # Dark energy density
        self.h_Hubble = 0.67
        self.cosmo = LambdaCDM(H0=self.h_Hubble * 100, Om0=OmegaCDM + OmegaBr, Ode0=OmegaLambda)

    def z_from_comoving_distance(self, distance, z_min, z_max):
        """
        Compute redshifts from an array of comoving distances using parallel processing.

        Parameters
        ----------
        distance : array-like
            Array of comoving distances in kpc.
        z_min : float
            Minimum redshift for search range.
        z_max : float
            Maximum redshift for search range.

        Returns
        -------
        np.ndarray
            Array of redshift values corresponding to the input distances.
        """
        max_ = len(distance)
        args = [(d, self.cosmo, z_min, z_max, self.h_Hubble) for d in distance]  # Argument tuples
        with Pool(processes=4) as p, tqdm(total=max_, mininterval=10) as pbar:
            result = []
            # Parallelized computation of redshifts
            for res in p.imap(compute_z, args):
                result.append(res)
                pbar.update()
            return np.array(result)

    def lens_weight(self, z_l, z_s=1100):
        """
        Compute the lensing weight function for a given lens redshift and source redshift(s).

        Parameters
        ----------
        z_l : float or array-like
            Lens redshift(s).
        z_s : float or array-like, optional
            Source redshift(s). Default is 1100 (CMB).

        Returns
        -------
        np.ndarray or float
            Lensing weight(s) for each lens redshift and source redshift pair.
        """
        d_comov_l = np.array(self.cosmo.comoving_distance(z_l))

        if np.isscalar(z_s):  # Single source redshift case
            if z_s == 0.:
                return 0.  # If the source redshift is 0, weight is 0
            d_comov_s = np.array(self.cosmo.comoving_distance(z_s))
            return (d_comov_s - d_comov_l) / d_comov_s

        # Handle array of source redshifts
        d_comov_s = np.array(self.cosmo.comoving_distance(z_s))

        # Safely compute weights, handling cases where `z_s` is zero
        np.seterr(invalid='ignore')
        lens_weights = np.where(np.isclose(d_comov_s, 0.), 0., (d_comov_s - d_comov_l) / d_comov_s)

        return lens_weights

    def conformal_hubble(self, z):
        """
        Compute the conformal Hubble parameter for a given redshift.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to compute the conformal Hubble parameter.

        Returns
        -------
        float or np.ndarray
            Conformal Hubble parameter value(s).
        """
        H_z = self.cosmo.H(z).value  # Hubble parameter in km/s/Mpc
        a_z = 1 / (1 + z)  # Scale factor
        mathcal_H = a_z * H_z  # Conformal Hubble parameter
        return mathcal_H


import numpy as np

class SnapsInfo:
    """
    A class to manage snapshot information from a text file and provide
    convenient access to data based on redshift (`z`) or snapshot index (`snap`).

    Attributes
    ----------
    columns : list of str
        List of column names in the data.
    data : dict
        Dictionary of data columns, where keys are column names and values
        are numpy arrays.
    z_index : dict
        Dictionary mapping redshift values (`z`) to their corresponding row index.
    snap_index : dict
        Dictionary mapping snapshot indices (`snap`) to their corresponding row index.
    """

    def __init__(self, path=""):
        """
        Initialize the SnapsInfo object by loading data from a file.

        Parameters
        ----------
        path : str, optional
            Path to the directory containing the `new_cosmoint_results.txt` file.
            Defaults to the current directory.
        """
        # Load data from the text file, skipping the header row
        data = np.loadtxt(f"{path}new_cosmoint_results.txt", skiprows=1)

        # Column names corresponding to the data
        self.columns = ['snap', 'z', 'distance', 'xstart', 'xend',
                        'zstart', 'zend', 'dx', 'boxes', 'cmb_weight']

        # Create a dictionary of data columns as numpy arrays
        self.data = {name: data[:, i] for i, name in enumerate(self.columns)}

        # Create dictionaries for quick access based on `z` and `snap` values
        self.z_index = {z: idx for idx, z in enumerate(self.data['z'])}
        self.snap_index = {snap: idx for idx, snap in enumerate(self.data['snap'])}

    def _validate_column(self, column_name):
        """
        Validate if the given column name exists in the data.

        Parameters
        ----------
        column_name : str
            Name of the column to validate.

        Raises
        ------
        ValueError
            If the column name is not valid.
        """
        if column_name not in self.columns:
            raise ValueError(f"Column '{column_name}' not found. Valid columns are: {self.columns}")

    def from_z_get(self, z_value, column_name):
        """
        Get the value of a specified column for a given redshift (`z`).

        Parameters
        ----------
        z_value : float
            The redshift value to look up.
        column_name : str
            Name of the column whose value is to be retrieved.

        Returns
        -------
        float
            The value of the specified column for the given redshift.

        Raises
        ------
        KeyError
            If the redshift value is not found.
        ValueError
            If the column name is invalid.
        """
        self._validate_column(column_name)
        idx = self.z_index.get(z_value)
        if idx is None:
            raise KeyError(f"z value {z_value} not found!")
        return self.data[column_name][idx]

    def from_snapshot_get(self, snap_value, column_name):
        """
        Get the value of a specified column for a given snapshot index (`snap`).

        Parameters
        ----------
        snap_value : float or int
            The snapshot value to look up.
        column_name : str
            Name of the column whose value is to be retrieved.

        Returns
        -------
        float
            The value of the specified column for the given snapshot.

        Raises
        ------
        KeyError
            If the snapshot value is not found.
        ValueError
            If the column name is invalid.
        """
        self._validate_column(column_name)
        idx = self.snap_index.get(snap_value)
        if idx is None:
            raise KeyError(f"snap value {snap_value} not found!")
        return self.data[column_name][idx]



class HaloInFile:
    """
    A helper class to define the data type for a simplified version of Halo objects
    when stored in a file.

    Methods
    -------
    get_dtype():
        Returns the numpy dtype describing the fields of the Halo data for file storage.
    """
    @classmethod
    def get_dtype(cls):
        """
        Defines the numpy dtype for halo data.

        Returns
        -------
        np.dtype
            A numpy dtype object describing the fields of a halo.
        """
        # Define the fields for the file representation of a halo
        fields = ['theta', 'phi', 'dr', 'mass', 'pos_x', 'pos_y', 'pos_z',
                  'vel_x', 'vel_y', 'vel_z']
        # Create a dtype object where each field is a double-precision float ('d')
        return np.dtype([(field, 'd') for field in fields])


class Halo:
    """
    A class representing a Halo object, including its position, velocity,
    physical properties, and cosmological data.

    Attributes
    ----------
    theta : float
        Angular coordinate theta of the halo in radians.
    phi : float
        Angular coordinate phi of the halo in radians.
    dr : float
        Radial comoving distance of the halo.
    mass : float
        Mass of the halo.
    position : tuple of float
        Cartesian coordinates (x, y, z) of the halo.
    vel : tuple of float
        Velocity components (v_x, v_y, v_z) of the halo.
    velocity : float
        Magnitude of the halo's velocity.
    v_parallel : float
        Component of the velocity parallel to the line of sight.
    pixel : int
        HEALPix pixel index for the halo's position on the sky.
    convergence : float
        Convergence value for the halo.
    luminosity_distance : float
        Luminosity distance to the halo.
    luminosity_distance_bert_convergence : float
        Luminosity distance corrected for Bertacca's convergence model.
    luminosity_distance_bert_v_parallel : float
        Luminosity distance corrected for Bertacca's velocity parallel model.
    distance_bertacca : float
        Distance correction according to Bertacca's model.
    redshift : float
        Redshift of the halo.

    Methods
    -------
    get_dtype():
        Returns the numpy dtype describing the fields of the Halo object.
    """
    def __init__(self, theta, phi, dr, mass,
                 pos_x, pos_y, pos_z,
                 vel_x, vel_y, vel_z, velocity, v_parallel,
                 pixel, convergence,
                 luminosity_distance, bertacca_convergence,
                 bertacca_v_parallel, distance_bertacca, redshift):
        self.theta = theta
        self.phi = phi
        self.dr = dr
        self.mass = mass
        self.position = (pos_x, pos_y, pos_z)
        self.vel = (vel_x, vel_y, vel_z)
        self.velocity = velocity
        self.v_parallel = v_parallel
        self.pixel = pixel
        self.convergence = convergence
        self.luminosity_distance = luminosity_distance
        self.luminosity_distance_bert_convergence = bertacca_convergence
        self.luminosity_distance_bert_v_parallel = bertacca_v_parallel
        self.distance_bertacca = distance_bertacca
        self.redshift = redshift

    @classmethod
    def get_dtype(cls):
        fields = [
            ('theta', 'd'),  # double
            ('phi', 'd'),  # double
            ('dr', 'd'),  # double
            ('mass', 'd'),  # double
            ('pos_x', 'd'),  # double
            ('pos_y', 'd'),  # double
            ('pos_z', 'd'),  # double
            ('vel_x', 'd'),  # double
            ('vel_y', 'd'),  # double
            ('vel_z', 'd'),  # double
            ('velocity', 'd'),  # double
            ('v_parallel', 'd'),  # double
            ('pixel', 'u4'),  # unsigned int (32-bit)
            ('convergence', 'd'),  # double
            ('luminosity_distance', 'd'), # double
            ('luminosity_distance_bert_convergence', 'd'), #double
            ('luminosity_distance_bert_v_parallel', 'd'),  # double
            ('luminosity_distance_bertacca', 'd'),  # double
            ('redshift', 'd')
        ]
        return np.dtype(fields)

class Catalog(Cosmo):
    """
    A class to handle halo catalogs from simulation snapshots, including various computations
    such as velocity, redshift, pixelization, and convergence properties.

    Attributes:
        snapshot_number (int): The snapshot number of the simulation.
        halos (np.ndarray): Array of halos with specific data types defined by Halo.get_dtype().
        path (str): Path to the directory containing the snapshot files.
        redshift (float): Redshift corresponding to the snapshot.
        z_start (float): Starting redshift of the snapshot range.
        z_end (float): Ending redshift of the snapshot range.
        n_halos (int): Total number of halos in the snapshot.
    """
    def __init__(self, snapshot_number, halos: Halo.get_dtype() = None, path = ""):
        """
        Initialize the Catalog class.

        Args:
            snapshot_number (int): The snapshot number of the simulation.
            halos (np.ndarray, optional): Array of halo data. If None, the halos are loaded from files.
            path (str, optional): Path to the directory containing snapshot files.
        """
        super().__init__()
        self.info = SnapsInfo(path=path)
        self.snapshot_number = snapshot_number
        self.redshift = self.info.from_snapshot_get(snapshot_number, 'z')
        self.z_start = self.info.from_snapshot_get(snapshot_number, 'zstart')
        self.z_end = self.info.from_snapshot_get(snapshot_number, 'zend')
        self.path = path

        if halos is None:
            self.halos = self.load_halos()
        else:
            if halos.dtype != Halo.get_dtype():
                raise ValueError("Invalid dtype for 'halos'")
            self.halos = halos
        self.n_halos = self.get_n_halos()



    def load_halos(self):
        """
        Load halo data from binary files for the current snapshot.

        Returns:
            np.ndarray: Array of halos with the correct data type.
        """
        files = [f"{self.path}AllSkyMock_snap_{self.snapshot_number:03d}_{i}.bin0" for i in range(4)]

        all_halos = None
        for file in files:
            if os.path.exists(file):
                print(f"Reading {file}...")
                halos_in_file = np.fromfile(file, dtype=HaloInFile.get_dtype())

                # Trasforma in un array con il dtype di Halo, includendo i nuovi campi
                halos = np.zeros(halos_in_file.shape, dtype=Halo.get_dtype())
                halos['theta'] = halos_in_file['theta']
                halos['phi'] = halos_in_file['phi']
                halos['dr'] = halos_in_file['dr']
                halos['mass'] = halos_in_file['mass']
                halos['pos_x'] = halos_in_file['pos_x']
                halos['pos_y'] = halos_in_file['pos_y']
                halos['pos_z'] = halos_in_file['pos_z']
                halos['vel_x'] = halos_in_file['vel_x']
                halos['vel_y'] = halos_in_file['vel_y']
                halos['vel_z'] = halos_in_file['vel_z']

                if all_halos is None:
                    all_halos = halos
                else:
                    all_halos = np.concatenate((all_halos, halos))
            else:
                print(f"File not found: {file}")

        return all_halos

    def get_n_halos(self):
        return len(self.halos) if self.halos is not None else 0

    def compute_velocity(self):
        # Extract velocity components
        vel_x, vel_y, vel_z = self.vel_x, self.vel_y, self.vel_z

        # Calculate total velocity
        velocity = np.sqrt(vel_x ** 2 + vel_y ** 2 + vel_z ** 2) * (1 + self.halos_redshift)

        # Calculate line-of-sight velocity
        theta = self.theta
        phi = self.phi
        v_parallel = vel_x * np.sin(theta) * np.cos(phi) + \
                     vel_y * np.sin(theta) * np.sin(phi) + \
                     vel_z * np.cos(theta)

        # Add computed velocities to halos
        self.halos['velocity'] = velocity
        self.halos['v_parallel'] = v_parallel

    def select_mass_above(self, mass_limit: float):
        """
        Filter the catalog to include halos with mass above a specified limit.

        Args:
            mass_limit (float): Minimum mass threshold.

        Returns:
            Catalog: A new Catalog instance with the filtered halos.
        """
        mask = self.mass > mass_limit
        filtered_halos = self.halos[mask]
        return Catalog(self.snapshot_number, filtered_halos, path = self.path)

    def compute_pixel(self, n_side = 4096):
        """
        Assign HEALPix pixel indices to each halo based on its angular coordinates.

        Args:
            n_side (int, optional): Resolution parameter for HEALPix. Default is 4096.
        """
        self.halos['pixel'] = hp.ang2pix(n_side, self.theta, self.phi)

    def compute_convergence(self, n_side: int = 4096):
        """
        Computes the convergence map for each snapshot in the simulation.

        For each snapshot, the method reads the convergence map from a FITS file, downgrades or raises an error
        based on the specified `n_side` resolution, and adds the contribution to the convergence field
        for the halos based on their lensing weight.

        Args:
            n_side (int): The resolution parameter for the healpy maps. Default is 4096.
        """
        for snapshot in range(62, self.snapshot_number - 1, -1):
            cmb_weight = self.info.from_snapshot_get(snapshot, 'cmb_weight')
            z_lens = self.info.from_snapshot_get(snapshot, 'z')

            print(f"-------------- {z_lens, snapshot} --------------")
            map_file = f'{self.path}KappaMap_snap_{snapshot:03d}.DM.seed_100672.fits'
            print(f"Reading {map_file}...")
            convergence_map = (1. / cmb_weight) * hp.read_map(map_file, dtype=np.float32)
            n_side_original_map = hp.get_nside(convergence_map)
            
            if n_side != n_side_original_map:
                if n_side < n_side_original_map:
                    print(f"Downgrading {map_file}...")
                    convergence_map = hp.ud_grade(convergence_map,  nside_out=n_side) #power??
                else:
                    raise ValueError("n_side for processing cannot exceed the original n_side of the map.")

            print(f"Calculating convergence contribution from {map_file}... \n")
            self.halos['convergence'] += self.lens_weight(z_lens, self.redshift) * convergence_map[self.pixel]

            del convergence_map

    def compute_redshift(self):
        """
        Computes the redshift of the halos based on their comoving distances.

        This method updates the 'redshift' field for all halos using the `z_from_comoving_distance` method.
        """
        self.halos['redshift'] = self.z_from_comoving_distance(self.halos['dr'], self.z_start, self.z_end)

    def compute_luminosity_distance(self):
        """
        Computes the luminosity distance for the halos based on their comoving distances and redshifts.

        The method updates the 'luminosity_distance' field for all halos using the formula:
            luminosity_distance = comoving_distance * (1 + redshift)
        """
        # Pre-calcolare la lista delle distanze comoventi
        #comoving_distances = self.halos['dr']

        # Calcolare i redshift solo una volta, evitando chiamate ripetitive
        #redshifts = self.z_from_comoving_distance(comoving_distances, self.z_start, self.z_end)

        # Calcolare la luminosity distance in modo vettoriale
        #self.halos['luminosity_distance'] = comoving_distances * (1 + redshifts)
        self.halos['luminosity_distance'] = self.halos['dr'] * (1 + self.halos['redshift'])

    def compute_bertacca_convergence(self):
        """
        Computes the Bertacca contribution to the luminosity distance due to convergence.

        This method updates the 'luminosity_distance_bert_convergence' field, which is the negative of
        the convergence.
        """
        self.halos['luminosity_distance_bert_convergence'] = (
                0 - self.halos['convergence']
        )

    def compute_bertacca_v_parallel(self):
        """
        Computes the Bertacca contribution to the luminosity distance due to parallel velocity.

        The method updates the 'luminosity_distance_bert_v_parallel' field based on the parallel velocity
        of the halos and their conformal Hubble parameter.
        """
        self.halos['luminosity_distance_bert_v_parallel'] = (
                 (1./astro.c.to(u.km / u.s).value - (1./(self.conformal_hubble(self.halos['redshift']) * self.halos['dr'])))* self.halos['v_parallel']
        )
        
    def compute_bertacca(self):
        """
        Computes the total Bertacca correction to the luminosity distance.

        This method updates the 'luminosity_distance_bertacca' field, which combines the original luminosity
        distance and the Bertacca convergence and velocity contributions.
        """
        self.halos['luminosity_distance_bertacca'] = (
                self.halos['luminosity_distance'] * (1 + self.halos['luminosity_distance_bert_convergence'] + self.halos['luminosity_distance_bert_v_parallel'])
        )

    def compute_all(self, n_side = 4096):
        """
        Computes all necessary quantities for the halos, including redshift, velocity, pixel,
        convergence, luminosity distance, Bertacca contributions, etc.

        The method checks if any of these fields are not yet computed and calculates them if necessary.

        Args:
            n_side (int): The resolution parameter for the healpy maps. Default is 4096.
        """
        if np.all(self.halos['redshift'] == 0.):
            self.compute_redshift()
        if np.all(self.halos['velocity'] == 0.):
            self.compute_velocity()
        if np.all(self.halos['pixel'] == 0):
            self.compute_pixel(n_side)
        if np.all(self.halos['convergence'] == 0.):
            self.compute_convergence(n_side)
        if np.all(self.halos['luminosity_distance'] == 0.):
            self.compute_luminosity_distance()
        if np.all(self.halos['luminosity_distance_bert_convergence'] == 0.):
            self.compute_bertacca_convergence()
        if np.all(self.halos['luminosity_distance_bert_v_parallel'] == 0.):
            self.compute_bertacca_v_parallel()
        if np.all(self.halos['luminosity_distance_bertacca'] == 0.):
            self.compute_bertacca()

    def halos_map(self, n_side: int = 4096):
        """
        Generates a map representing the halos based on their pixel assignments.

        Args:
            n_side (int): The resolution parameter for the healpy map. Default is 4096.

        Returns:
            np.ndarray: The halo map with the number of halos in each pixel.
        """
        halos_map = np.zeros(hp.nside2npix(n_side))
        np.add.at(halos_map, self.halos['pixel'], 1)
        return halos_map

    def bertacca_map(self, n_side: int = 4096):
        """
        Generates a Bertacca map based on the luminosity distance corrections.

        Args:
            n_side (int): The resolution parameter for the healpy map. Default is 4096.

        Returns:
            np.ndarray: The Bertacca luminosity distance map with averaged values per pixel.
        """
        # Ottieni il numero totale di pixel per il dato n_side
        npix = hp.nside2npix(n_side)

        # Array per i valori accumulati
        total_values = np.zeros(npix)
        # Array per contare quante volte ogni pixel è stato toccato
        counts = np.zeros(npix)

        # Accumula i valori e conta le occorrenze per ogni pixel
        np.add.at(total_values, self.halos['pixel'], self.halos['luminosity_distance_bertacca'])
        np.add.at(counts, self.halos['pixel'], 1)

        # Calcola la media per i pixel con almeno un valore
        non_zero_mask = counts > 0
        bertacca_map = np.zeros(npix)
        bertacca_map[non_zero_mask] = total_values[non_zero_mask] / counts[non_zero_mask]

        # Optional: stampa per segnalare pixel duplicati
        duplicates = np.sum(counts > 1)
        if duplicates > 0:
            print(f"Same pixel encountered {duplicates} times.")
        return bertacca_map

    @property
    def theta(self):
        return self.halos['theta']

    @property
    def phi(self):
        return self.halos['phi']

    @property
    def mass(self):
        return self.halos['mass']

    @property
    def distance(self):
        return self.halos['dr']

    @property
    def pos_x(self):
        return self.halos['pos_x']

    @property
    def pos_y(self):
        return self.halos['pos_y']

    @property
    def pos_z(self):
        return self.halos['pos_z']

    @property
    def vel_x(self):
        return self.halos['vel_x']

    @property
    def vel_y(self):
        return self.halos['vel_y']

    @property
    def vel_z(self):
        return self.halos['vel_z']

    @property
    def velocity(self):
        return self.halos['velocity']

    @property
    def v_parallel(self):
        return self.halos['v_parallel']

    @property
    def pixel(self):
        return self.halos['pixel']

    @property
    def convergence(self):
        return self.halos['convergence']

    @property
    def luminosity_distance(self):
        return self.halos['luminosity_distance']

    @property
    def bertacca_convergence(self):
        return self.halos['luminosity_distance_bert_convergence']

    @property
    def bertacca_v_parallel(self):
        return self.halos['luminosity_distance_bert_v_parallel']

    @property
    def bertacca(self):
        return self.halos['luminosity_distance_bertacca']

    @property
    def halos_redshift(self):
        return self.halos['redshift']

    def get_halo(self, i):
        return Halo(self.theta[i], self.phi[i], self.distance[i],
                    self.mass[i], self.pos_x[i], self.pos_y[i], self.pos_z[i],
                    self.vel_x[i], self.vel_y[i], self.vel_z[i], self.velocity[i], self.v_parallel[i],
                    self.pixel[i], self.convergence[i],
                    self.luminosity_distance[i], self.bertacca_convergence[i],
                    self.bertacca_v_parallel[i], self.bertacca[i], self.halos_redshift[i])