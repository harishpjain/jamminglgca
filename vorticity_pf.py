import numpy as np
from scipy.ndimage.filters import gaussian_filter1d, uniform_filter1d
from lgca.lgca_hex import *

path = '/mnt/data/simon/friedl_paper_raw_data/friedl_project/final/'

ecm_densities = np.load(path + 'densities.npy')
betas = np.load(path + 'betas.npy')

vorticity = np.zeros((len(ecm_densities), len(betas)))
nb_correlation = vorticity.copy()
speed = vorticity.copy()
speed_const_window = speed.copy()
speed_gauss_window = speed.copy()

i, j = 0, 0
for ecm_density in np.round(ecm_densities[:], decimals=2):
    j = 0
    for beta in np.round(betas[:], decimals=2):
        nodes_nt = np.load(path + 'nodes_nt_density_{}_beta_{}.npy'.format(ecm_density, beta))
        if i == 0 and j == 0:
            n, timesteps, lx, ly, K = nodes_nt.shape
            lgca = LGCA_Hex(lx=lx, ly=ly, restchannels=K-6)

        nodes_nt = np.moveaxis(nodes_nt, 2, 0)
        nodes_nt = np.moveaxis(nodes_nt, 3, 1)
        occupied = nodes_nt.astype(bool)
        flux = lgca.calc_flux(occupied)
        dens = occupied.sum(-1)
        v = np.divide(flux, dens[..., None], where=dens[..., None] > 0, out=np.zeros_like(flux))
        v_norm = np.linalg.norm(v, axis=-1)

        #speed[i, j] = np.mean(v_norm[dens > 0])

        #averaged_v_1d = uniform_filter1d(v, 5, axis=3)
        #uni_v_norm = np.linalg.norm(averaged_v_1d, 5, axis=-1)
        #uni_dens = uniform_filter1d(dens, 5, axis=3)
        #speed_const_window[i, j] = np.mean(uni_v_norm[uni_dens > 0])

        #averaged_v_gauss = gaussian_filter1d(v, 5, axis=3)
        #gauss_v_norm = np.linalg.norm(averaged_v_gauss, axis=-1)
        #gauss_dens = gaussian_filter1d(dens, 5, axis=3)
        #speed_gauss_window[i, j] = np.mean(gauss_v_norm[gauss_dens > 0])

        # fx, fy = flux[..., 0], flux[..., 1]
        # dfx = lgca.gradient(fx)
        # dfy = lgca.gradient(fy)
        # dfxdy = dfx[..., 1]
        # dfydx = dfy[..., 0]
        # vorticity_field = abs(dfydx - dfxdy)
        # vorticity[i, j] = np.mean(vorticity_field[dens > 0])
        nb_v = lgca.nb_sum(v)
        # nb_fy = lgca.nb_sum(fy)
        corr = np.einsum('...i,...i', v, nb_v)
        nb_v_norm = np.linalg.norm(nb_v, axis=-1)
        corr = np.divide(corr, v_norm, where=v_norm > 0, out=np.zeros_like(corr))
        corr = np.divide(corr, nb_v_norm, where=nb_v_norm > 0, out=np.zeros_like(corr))
        nb_correlation[i, j] = np.mean(corr[dens > 0])
        # break
        j += 1
    i += 1
    # break

# np.save(path + 'corrected_vorticity.npy', vorticity)
#np.save(path + 'mean_corr.npy', nb_correlation)
#np.save(path + 'mean_speed.npy', speed)
#np.save(path + 'mean_speed_const_window.npy', speed_const_window)
#np.save(path + 'mean_speed_gauss_window.npy', speed_gauss_window)




