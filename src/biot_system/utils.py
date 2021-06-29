import matplotlib.pyplot as plt
import scipy.sparse as sparse
import fenics as pde
import numpy as np


def petsc_to_scipy(petsc_matrix):
    return sparse.csr_matrix(petsc_matrix.getValuesCSR()[::-1])


def dolfin_to_scipy(dolfin_matrix):
    return petsc_to_scipy(pde.as_backend_type(dolfin_matrix).mat())


def fenics_quiver(F, mesh_size=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if mesh_size is None:
        mesh_size = F.function_space().mesh().coordinates().max(axis=0)

    x = np.linspace(0, mesh_size[0], 20)
    y = np.linspace(0, mesh_size[1], 20)
    xx, yy = np.meshgrid(x, y)
    uu = np.empty_like(xx).ravel()
    vv = np.empty_like(xx).ravel()
    for i, (x, y) in enumerate(zip(xx.ravel(), yy.ravel())):
        uu[i] = F(x, y)[0]
        vv[i] = F(x, y)[1]

    ax.quiver(
        xx,
        yy,
        uu.reshape(xx.shape),
        vv.reshape(xx.shape),
        np.sqrt(uu**2 + vv**2).reshape(xx.shape),
        cmap="magma",
    )
    ax.set_title(f"U: {uu.max():.1g}, {uu.min():.1g}, V: {vv.max():.1g}, {vv.min():.1g}")
    ax.axis("equal")


def fenics_contour(F, mesh_size=None, ax=None, clims=None):
    if ax is None:
        ax = plt.gca()
    if mesh_size is None:
        mesh_size = F.function_space().mesh().coordinates().max(axis=0)

    x = np.linspace(0, mesh_size[0], 100)
    y = np.linspace(0, mesh_size[1], 100)
    xx, yy = np.meshgrid(x, y)
    zz = np.empty_like(xx).ravel()
    for i, (x, y) in enumerate(zip(xx.ravel(), yy.ravel())):
        zz[i] = F((x, y))

    if clims is None:
        vmax = np.max(np.abs(zz))
        vmin = -vmax
    else:
        vmin, vmax = clims
    ax.contourf(xx, yy, zz.reshape(xx.shape), vmin=vmin, vmax=vmax, cmap="coolwarm", levels=100)
    ax.set_title(f"{zz.max():1f}, {zz.min():.1f}")
    ax.axis("equal")