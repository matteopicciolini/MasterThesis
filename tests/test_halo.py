from halo import AllHalosFromCatalogs
import healpy as hp
import numpy as np


def test_update_convergence_halos():
    phi = np.array([1.18217856, 0.63124405, 4.83473766])
    theta = np.array([2.38734685, 2.58877644, 2.7756767])
    pixels = np.array([174015932, 186328928, 194660008])
    snapshots = np.array([48, 49, 50])
    redshifts = np.array([0.000000, 0.074610, 0.485510])
    data = AllHalosFromCatalogs(theta, phi, snapshots, redshifts, range(48, 51), path="../")

    assert (data.pixels == hp.ang2pix(4096, theta, phi)).all()
    assert (pixels == data.pixels).all()

    map_file_50 = f'../KappaMap_snap_050.DM.seed_100672.fits'
    convergence_map_50 = hp.read_map(map_file_50, dtype=np.float32)
    map_file_49 = f'../KappaMap_snap_049.DM.seed_100672.fits'
    convergence_map_49 = hp.read_map(map_file_49, dtype=np.float32)
    map_file_48 = f'../KappaMap_snap_048.DM.seed_100672.fits'
    convergence_map_48 = hp.read_map(map_file_48, dtype=np.float32)

    assert (convergence_map_50[194660008] == data.convergence_halos[194660008])
    assert (convergence_map_50[186328928] + convergence_map_49[186328928] == data.convergence_halos[186328928])
    assert (convergence_map_50[174015932] + convergence_map_49[174015932] + convergence_map_48[174015932] ==
            data.convergence_halos[174015932])
    assert (data.convergence_halos[0] == 0.)


def test_update_convergence_halos_2():
    phi = np.array([1.18217856, 0.63124405, 4.83473766, 5.82791877, 1.14823037])
    theta = np.array([2.38734685, 2.58877644, 2.7756767,  2.03436287, 2.666516])
    pixels = np.array([174015932, 186328928, 194660008, 145677149, 190174953])
    snapshots = np.array([59, 61, 61, 62, 62])
    redshifts = np.array([0.48551, 0.48551, 0.48551, 0.48551, 0.48551])
    data = AllHalosFromCatalogs(theta, phi, snapshots, redshifts, range(59, 63), path="../")

    assert (data.pixels == hp.ang2pix(4096, theta, phi)).all()
    assert (pixels == data.pixels).all()

    map_file_62 = f'../KappaMap_snap_062.DM.seed_100672.fits'
    convergence_map_62 = hp.read_map(map_file_62, dtype=np.float32)
    map_file_61 = f'../KappaMap_snap_061.DM.seed_100672.fits'
    convergence_map_61 = hp.read_map(map_file_61, dtype=np.float32)
    map_file_60 = f'../KappaMap_snap_060.DM.seed_100672.fits'
    convergence_map_60 = hp.read_map(map_file_60, dtype=np.float32)
    map_file_59 = f'../KappaMap_snap_059.DM.seed_100672.fits'
    convergence_map_59 = hp.read_map(map_file_59, dtype=np.float32)

    assert (convergence_map_62[190174953] == data.convergence_halos[190174953])
    assert (convergence_map_62[145677149] == data.convergence_halos[145677149])
    assert (convergence_map_62[194660008] + convergence_map_61[194660008]  == data.convergence_halos[194660008])
    assert (convergence_map_62[186328928] + convergence_map_61[186328928] == data.convergence_halos[186328928])
    assert (convergence_map_62[174015932] + convergence_map_61[174015932] + convergence_map_60[174015932] + convergence_map_59[174015932] == data.convergence_halos[174015932])
    assert (data.convergence_halos[0] == 0.)
