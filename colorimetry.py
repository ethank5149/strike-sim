"""
colorimetry.py -- Spectral radiometry for the channel and its glow.

The return-stroke channel is a ~30,000 K plasma; its color follows from
integrating Planck's law against the CIE 1931 color-matching functions
(multi-Gaussian analytic fits of Wyman, Sloan & Shirley, JCGT 2(2), 2013,
accurate to ~1%), then converting XYZ -> linear sRGB (IEC 61966-2-1
matrix, D65 white).

The halo around the channel is light scattered by air molecules and
droplets; the molecular (Rayleigh) component has cross-section ~ 1/l^4,
so the glow is weighted bluer than the direct channel light.
"""

import numpy as np

# Planck constants (SI)
_H = 6.62607015e-34
_C = 2.99792458e8
_KB = 1.380649e-23


def _gauss(x, alpha, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return alpha * np.exp(-0.5 * ((x - mu) / s) ** 2)


def cie_xyz_bar(lam_nm):
    """CIE 1931 2-deg color matching functions (Wyman et al. 2013 fits)."""
    x = (_gauss(lam_nm, 1.056, 599.8, 37.9, 31.0)
         + _gauss(lam_nm, 0.362, 442.0, 16.0, 26.7)
         + _gauss(lam_nm, -0.065, 501.1, 20.4, 26.2))
    y = (_gauss(lam_nm, 0.821, 568.8, 46.9, 40.5)
         + _gauss(lam_nm, 0.286, 530.9, 16.3, 31.1))
    z = (_gauss(lam_nm, 1.217, 437.0, 11.8, 36.0)
         + _gauss(lam_nm, 0.681, 459.0, 26.0, 13.8))
    return x, y, z


def planck(lam_m, T):
    """Spectral radiance B_lambda(T), W sr^-1 m^-3."""
    a = 2.0 * _H * _C ** 2 / lam_m ** 5
    b = _H * _C / (lam_m * _KB * T)
    return a / np.expm1(b)


_XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                         [-0.9689, 1.8758, 0.0415],
                         [0.0557, -0.2040, 1.0570]])


def spectrum_to_linear_srgb(lam_nm, S):
    xb, yb, zb = cie_xyz_bar(lam_nm)
    X = np.trapezoid(S * xb, lam_nm)
    Y = np.trapezoid(S * yb, lam_nm)
    Z = np.trapezoid(S * zb, lam_nm)
    rgb = _XYZ_TO_SRGB @ np.array([X, Y, Z])
    return rgb, (X, Y, Z)


def blackbody_rgb(T, rayleigh=False, norm="maxone"):
    """Linear-sRGB color of a blackbody at T kelvin.

    rayleigh=True applies the lambda^-4 molecular-scattering weight
    (color of the scattered halo rather than the direct channel light).
    """
    lam_nm = np.linspace(380.0, 780.0, 401)
    S = planck(lam_nm * 1e-9, T)
    if rayleigh:
        S = S * (550.0 / lam_nm) ** 4
    rgb, XYZ = spectrum_to_linear_srgb(lam_nm, S)
    rgb = np.clip(rgb, 0.0, None)
    if norm == "maxone":
        rgb = rgb / rgb.max()
    return rgb, XYZ


def chromaticity(XYZ):
    X, Y, Z = XYZ
    s = X + Y + Z
    return X / s, Y / s
