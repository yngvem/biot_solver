import matplotlib.pyplot as plt
import scipy.sparse as sparse
import fenics as pde
import numpy as np
import matplotlib.colors as mcolor
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable



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
    norm = mcolor.Normalize()
    norm.autoscale(np.sqrt(uu**2 + vv**2))
    sm = cm.ScalarMappable(cmap="magma", norm=norm)

    #ax.set_title(f"U: {uu.max():.1g}, {uu.min():.1g}, V: {vv.max():.1g}, {vv.min():.1g}")
    ax.axis("equal")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.figure.colorbar(sm, cax=cax, orientation='vertical')
    ax.set_yticks([])
    ax.set_xticks([])


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
    img = ax.contourf(xx, yy, zz.reshape(xx.shape), vmin=vmin, vmax=vmax, cmap="coolwarm", levels=100)
    #ax.set_title(f"{zz.max():.1g}, {zz.min():.1g}")
    ax.axis("equal")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.figure.colorbar(img, cax=cax, orientation='vertical')
    ax.set_yticks([])
    ax.set_xticks([])


def plot_solution(true_u, true_p, estimated_u, estimated_p, V, Q):
    fig, axes = plt.subplots(3, 2, dpi=200, tight_layout=True)
    fenics_quiver(true_u, ax=axes[0, 0])
    fenics_quiver(estimated_u, ax=axes[1, 0])
    fenics_quiver(pde.project(true_u - estimated_u, V), ax=axes[2, 0])
    axes[0, 0].set_ylabel("True u")
    axes[1, 0].set_ylabel("Estimated u")
    axes[2, 0].set_ylabel("Error u")

    fenics_contour(true_p, ax=axes[0, 1])
    fenics_contour(estimated_p, ax=axes[1, 1])
    fenics_contour(pde.project(true_p - estimated_p, Q), ax=axes[2, 1])
    axes[0, 1].set_ylabel("True pF")
    axes[1, 1].set_ylabel("Estimated pF")
    axes[2, 1].set_ylabel("Error pF")
    return fig, axes
