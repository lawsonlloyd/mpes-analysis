#fake_data.py

import numpy as np
import xarray as xr

def make_fake_trarpes_data():
    # axes
    kx = np.linspace(-2, 2, 80)
    ky = np.linspace(-2, 2, 80)
    E = np.linspace(-4, 2, 120)
    delay = np.linspace(-250, 1000, 100)

    KX, KY, EE = np.meshgrid(kx, ky, E, indexing="ij")

    # ---------------------------
    # fake band structure
    # ---------------------------
    band = np.exp(-(KX**2 + KY**2) / 0.6)

    # energy distribution (centered near E=0)
    spectral = np.exp(-((EE) / 0.25) ** 2)

    # ---------------------------
    # pump / time dynamics
    # ---------------------------
    D = delay

    # simple rise + decay (like exciton → carrier transfer)
    dynamics = np.exp(-((D - 200) / 250) ** 2) + 0.3 * np.exp(-(D + 100) ** 2 / 800)

    # reshape for broadcasting
    band = band[:, :, :, None]
    spectral = spectral[:, :, :, None]
    dynamics = dynamics[None, None, None, :]

    # full dataset
    I = band * spectral * dynamics

    # add noise (important for realism)
    I += 0.05 * np.random.rand(*I.shape)

    # wrap in xarray
    data = xr.DataArray(
        I,
        dims=("kx", "ky", "E", "delay"),
        coords={
            "kx": kx,
            "ky": ky,
            "E": E,
            "delay": delay,
        },
        name="intensity",
    )

    return data