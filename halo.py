import numpy as np
import os

class HaloCatalog:
    # Definire il tipo di dati binari
    dtype_halo = np.dtype([
        ('theta', 'd'), ('phi', 'd'), ('dr', 'd'),
        ('Mass', 'd'), ('pos_x', 'd'), ('pos_y', 'd'),
        ('pos_z', 'd'), ('vel_x', 'd'), ('vel_y', 'd'),
        ('vel_z', 'd')
    ])
    
    def __init__(self, snapshot_number, halos=None):
        self.snapshot_number = snapshot_number
        if halos is not None:
            self.halos = halos
        else:
            self.halos = self.load()
        self.n_halos = self.count()

    def read_halos(self, file_name):
        """Legge i dati binari di un singolo file"""
        return np.fromfile(file_name, dtype=self.dtype_halo)

    def load(self):
        """Carica i dati degli aloni combinando i file binari per lo snapshot"""
        files = [f"AllSkyMock_snap_{self.snapshot_number:03d}_{i}.bin0" for i in range(4)]


        all_halos = []

        for file in files:
            if os.path.exists(file):
                print(f"Reading {file}")
                halos = self.read_halos(file)
                all_halos.append(halos)
            else:
                print(f"File not found: {file}")


        return np.concatenate(all_halos)
    
    
    def count(self):
        """Restituisce il numero di aloni in self.halos"""
        return len(self.halos)

    @property
    def theta(self):
        """Ritorna i valori di theta per tutti gli aloni"""
        return self.halos['theta']

    @property
    def phi(self):
        """Ritorna i valori di phi per tutti gli aloni"""
        return self.halos['phi']

    @property
    def mass(self):
        """Ritorna le masse degli aloni"""
        return self.halos['Mass']
    
    def select_mass_above(self, mass_limit: float) -> 'HaloCatalog':
        """Filtra gli aloni in base a un limite di massa e restituisce una nuova istanza"""
        mask = self.halos['Mass'] > mass_limit
        filtered_halos = self.halos[mask]
        return HaloCatalog(self.snapshot_number, filtered_halos)
    
    
    def calculate_velocity(self) -> np.ndarray:
        """Calcola il vettore velocità per ogni alone"""
        vel_x = self.halos['vel_x']
        vel_y = self.halos['vel_y']
        vel_z = self.halos['vel_z']
        velocities = np.vstack((vel_x, vel_y, vel_z)).T  
        return np.linalg.norm(velocities, axis=1)
    

class HaloCatalogManager:
    def __init__(self, snapshot_numbers, catalogs = None):
        if catalogs is not None:
            self.snapshot_numbers = snapshot_numbers
            self.catalogs = catalogs
        else:
            self.snapshot_numbers = snapshot_numbers
            self.catalogs = [HaloCatalog(snapshot_number) for snapshot_number in snapshot_numbers]
        
        
    def get_combined_halos(self):
        """Combina gli aloni da tutti i cataloghi"""
        all_halos = np.concatenate([catalog.halos for catalog in self.catalogs])
        return all_halos

    def select_mass_above(self, mass_limit: float) -> 'HaloCatalogManager':
        """Filtra gli aloni in tutti i cataloghi in base a un limite di massa e restituisce un nuovo HaloCatalogManager"""
        filtered_catalogs = [catalog.select_mass_above(mass_limit) for catalog in self.catalogs]
        
        return HaloCatalogManager(self.snapshot_numbers, catalogs=filtered_catalogs)
    
    def calculate_velocity(self) -> list[np.ndarray]:
        """Calcola la velocità per ogni catalogo e ritorna una lista delle velocità"""
        return [catalog.calculate_velocity() for catalog in self.catalogs]
    
    @property
    def theta(self) -> list[np.ndarray]:
        return [catalog.theta for catalog in self.catalogs]
    
    @property
    def mass(self) -> list[np.ndarray]:
        return [catalog.mass for catalog in self.catalogs]
    
    @property
    def phi(self) -> list[np.ndarray]:
        return [catalog.phi for catalog in self.catalogs]