from unittest import TestCase
from halo import *
import healpy as hp
import numpy as np


class TestAllHalosFromCatalogs(TestCase):

    def test_update_convergence_halos(self):
        phi = np.array([1.18217856, 0.63124405, 4.83473766])
        theta = np.array([2.38734685, 2.58877644, 2.7756767])
        pixels = np.array([174015932, 186328928, 194660008])
        snapshots = np.array([48, 49, 50])
        redshifts = np.array([0.485510, 0.433010, 0.382370])




        data = Halos(theta, phi, snapshots, redshifts, range(48, 51), path="../")

        assert (data.pixels == hp.ang2pix(4096, theta, phi)).all()
        assert (pixels == data.pixels).all()

        map_file_50 = f'../KappaMap_snap_050.DM.seed_100672.fits'
        convergence_map_50 = hp.read_map(map_file_50, dtype=np.float32)
        map_file_49 = f'../KappaMap_snap_049.DM.seed_100672.fits'
        convergence_map_49 = hp.read_map(map_file_49, dtype=np.float32)
        map_file_48 = f'../KappaMap_snap_048.DM.seed_100672.fits'
        convergence_map_48 = hp.read_map(map_file_48, dtype=np.float32)

        cmb_weight_48 = 1. / lens_weight(0.485510)
        cmb_weight_49 = 1. / lens_weight(0.433010)
        cmb_weight_50 = 1. / lens_weight(0.382370)

        lens_weight_50_48 = lens_weight(0.382370, 0.485510)
        lens_weight_49_48 = lens_weight(0.433010, 0.485510)
        lens_weight_48_48 = lens_weight(0.485510, 0.485510)
        lens_weight_50_49 = lens_weight(0.382370, 0.433010)
        lens_weight_49_49 = lens_weight(0.433010, 0.433010)
        lens_weight_50_50 = lens_weight(0.382370, 0.382370)

        first = (lens_weight_50_48 * cmb_weight_50 * convergence_map_50[174015932] +
                 lens_weight_49_48 * cmb_weight_49 * convergence_map_49[174015932] +
                 lens_weight_48_48 * cmb_weight_48 * convergence_map_48[174015932])
        second = (lens_weight_50_49 * cmb_weight_50 * convergence_map_50[186328928] +
                  lens_weight_49_49 * cmb_weight_49 * convergence_map_49[186328928])
        third = lens_weight_50_50 * cmb_weight_50 * convergence_map_50[194660008]

        assert np.isclose(first, data.convergence[0], rtol=1e-7)
        assert np.isclose(second, data.convergence[1], rtol=1e-7)
        assert np.isclose(third, data.convergence[2], rtol=1e-7)



    def test_update_convergence_halos_2(self):
        phi = np.array([1.18217856, 0.63124405, 4.83473766, 5.82791877, 1.14823037])
        theta = np.array([2.38734685, 2.58877644, 2.7756767, 2.03436287, 2.666516])
        pixels = np.array([174015932, 186328928, 194660008, 145677149, 190174953])
        snapshots = np.array([59, 61, 61, 62, 62])
        redshifts = np.array([0.074610, 0.018150, 0.018150, 0.000000, 0.000000])
        data = Halos(theta, phi, snapshots, redshifts, range(59, 63), path="../")

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

        cmb_weight_62 = 1. / lens_weight(0.)
        cmb_weight_61 = 1. / lens_weight(0.018150)
        cmb_weight_60 = 1. / lens_weight(0.055450)
        cmb_weight_59 = 1. / lens_weight(0.074610)

        lens_weight_62_62 = lens_weight(0., 0.)
        lens_weight_62_61 = lens_weight(0., 0.018150)
        lens_weight_61_61 = lens_weight(0.018150, 0.018150)
        lens_weight_62_59 = lens_weight(0., 0.074610)
        lens_weight_61_59 = lens_weight(0.018150, 0.074610)
        lens_weight_60_59 = lens_weight(0.055450, 0.074610)
        lens_weight_59_59 = lens_weight(0.074610, 0.074610)


        assert np.isclose(lens_weight_62_59 * cmb_weight_62 * convergence_map_62[174015932] +
                          lens_weight_61_59 * cmb_weight_61 * convergence_map_61[174015932] +
                          lens_weight_60_59 * cmb_weight_60 * convergence_map_60[174015932] +
                          lens_weight_59_59 * cmb_weight_59 * convergence_map_59[174015932],
                          data.convergence[0], rtol=1e-7)

        assert (lens_weight_62_61 * cmb_weight_62 * convergence_map_62[186328928] +
                lens_weight_61_61 * cmb_weight_61 * convergence_map_61[186328928] == data.convergence[1])

        assert (lens_weight_62_61 * cmb_weight_62 * convergence_map_62[194660008] +
                lens_weight_61_61 * cmb_weight_61 * convergence_map_61[194660008] == data.convergence[2])

        assert (lens_weight_62_62 * cmb_weight_62 * convergence_map_62[145677149] == data.convergence[3])

        assert np.isclose(lens_weight_62_62 * cmb_weight_62 * convergence_map_62[190174953],
                          data.convergence[4], rtol=1e-7)


