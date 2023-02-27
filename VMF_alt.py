import numpy as np


def sizeToShape(size):
    isNone, isInt = (size is None), isinstance(size, int)
    shape = () if isNone else (size,) if isInt else size
    return tuple(shape)


"""
This version is faster than that of this reference tutorial with which we started:
    https://dlwhittenbury.github.io/ds-2-sampling-and-visualising-the-von-mises-fisher-distribution-in-p-dimensions.html    
"""


def random_VMF(mu, kappa, size=None):
    """
    Implementation of the von Mises-Fisher distribution with
    mean direction mu and concentration kappa.

    @author: Carlos Pinzón caph1993@gmail.com
    @author: Kangsoo Jung
    """
    # parse input parameters
    shape = () if size is None else tuple(np.ravel(size))
    n = 1 if size is None else np.product(size)
    mu = np.asarray(mu)
    mu = mu / np.linalg.norm(mu)
    (d,) = mu.shape
    # z component: radial samples perpendicular to mu
    z = np.random.normal(0, 1, (n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z = z - (z @ mu[:, None]) * mu[None, :]
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    # sample angles (in cos and sin form)
    cos = _random_VMF_cos(d, kappa, n)
    sin = np.sqrt(1 - cos**2)
    # combine angles with the z component
    x = z * sin[:, None] + cos[:, None] * mu[None, :]
    return x.reshape((*shape, d))


def _random_VMF_cos(d: int, kappa: float, n: int):
    """
    Generate n iid samples t with density function given by
      p(t) = someConstant * (1-t**2)**((d-2)/2) * exp(kappa*t)
    """
    # b = Eq. 4 of https://doi.org/10.1080/03610919408813161
    b = (d - 1) / (2 * kappa + (4 * kappa**2 + (d - 1) ** 2) ** 0.5)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0**2)
    found = 0
    out = []
    while found < n:
        m = min(n, int((n - found) * 1.5))
        z = np.random.beta((d - 1) / 2, (d - 1) / 2, size=m)
        t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        test = kappa * t + (d - 1) * np.log(1 - x0 * t) - c
        accept = test >= -np.random.exponential(size=m)
        out.append(t[accept])
        found += len(out[-1])
    return np.concatenate(out)[:n]


def sample_theta(d: int, kappa: float, n: int):
    """
    Generate n iid samples theta with density function given by
        p(theta) = someConstant * sin(theta)**d * exp(kappa*cos(theta))
    """
    # b = Eq. 4 of https://doi.org/10.1080/03610919408813161
    b = (d - 1) / (2 * kappa + np.sqrt(4 * kappa**2 + (d - 1) ** 2))
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0**2)
    end = 0
    out = np.zeros(n)
    while end < n:
        Z = np.random.beta((d - 1) / 2, (d - 1) / 2, size=n)
        U = np.random.random(size=n)
        cos = (1 - (1 + b) * Z) / (1 - (1 - b) * Z)
        acc = kappa * cos + (d - 1) * np.log(1 - x0 * cos) - c >= np.log(U)
        out[end : end + len(cos)] = cos[acc][: n - end]
    theta = np.arccos(out)
    return theta


import timeit


def _acceptance_test(d, kappa, n, repeat=10):
    out = []
    f = lambda: out.append(__acceptance_test(d, kappa, n))
    t = timeit.timeit(f, number=repeat) / repeat
    acc = np.mean(out)
    return acc, t


def __acceptance_test(d, kappa, n):
    b = (d - 1) / (2 * kappa + np.sqrt(4 * kappa**2 + (d - 1) ** 2))
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0**2)
    Z = np.random.beta((d - 1) / 2, (d - 1) / 2, size=n)
    U = np.random.random(size=n)
    cos = (1 - (1 + b) * Z) / (1 - (1 - b) * Z)
    accept = kappa * cos + (d - 1) * np.log(1 - x0 * cos) - c >= np.log(U)
    return accept.sum() / n


def sample_theta2(d, kappa, n):
    f = lambda theta: np.sin(theta) ** (d - 2) * np.exp(kappa * np.cos(theta))
    randTheta = lambda n: np.pi * (np.random.random(n) * 2 - 1)
    vMax = f(theta_arg_max(d, kappa))
    out = []
    while len(out) < n:
        theta = randTheta(n)
        y = f(theta)
        out.extend(theta[np.random.random(n) * vMax < y])
    out = np.array(out[:n])
    return out


def theta_arg_max(d, kappa):
    maxCos = ((d - 2) - np.sqrt((d - 2) ** 2 + 4 * kappa**2)) / (2 * kappa)
    mleTheta = np.arccos(np.abs(maxCos))
    assert np.isfinite(mleTheta)
    return mleTheta
