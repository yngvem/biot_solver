import argparse

import fenics as pde
import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from biot_system.utils import dolfin_to_scipy
from biot_system import BiotSystem


#%% Setup arguments
parser = argparse.ArgumentParser()
parser.add_argument("--no-exact", dest="exact", action='store_false')
parser.add_argument("--no-approx", dest="exact", action='store_false')
parser.set_defaults(exact=True, approx=True)
args = parser.parse_args()
compute_condition_number = args.exact
compute_approximate_condition_number = args.approx


#%% Setup mesh
condition_numbers = {'Num timesteps': [], 'Spatial density': []}
if compute_condition_number:
    condition_numbers['Condition number (exact)'] = []
if compute_approximate_condition_number:
    condition_numbers['Condition number (approximate)'] = []


for nT in [1, 32, 32*32, 32*32*32]:
    for nX in [4, 8, 16, 32]:
        print("nT", nT, "nX", nX)
        condition_numbers['Num timesteps'].append(nT)
        condition_numbers['Spatial density'].append(nX)

        mesh = pde.UnitSquareMesh(nX, nX)

        #%% Physical parameters
        zero = pde.Constant(0)
        zero_vector = pde.Constant((0, 0))
        mu = pde.Constant(0.5)
        lambda_ = pde.Constant(1)
        alpha = pde.Constant(1)
        kappa = pde.Constant(1)
        dt = pde.Constant(1/nT)
        f = zero_vector
        g = zero

        #%% Setup bilinear form
        system = BiotSystem(
            mesh=mesh,
            shear_modulus=mu,
            lame_parameter=lambda_,
            biot_coefficient=alpha,
            hydraulic_conductivity=kappa,
            force_field=f,
            source_field=g,
            dt=dt,
        )

        solution = system.solution
        previous_solution = system.previous_solution
        u, pT, pF = system.trial_functions
        v, qT, qF = system.test_functions

        a = system.get_lhs()
        preconditioner = system.get_preconditioner()

        #%% Setup BCs
        bcs = [
            pde.DirichletBC(system.function_space.sub(0), zero_vector, "on_boundary"),
            pde.DirichletBC(system.function_space.sub(2), zero, "on_boundary"),
        ]

        #%% Main task
        print("Assembling matrices and applying boundary condition")
        A = pde.assemble(a)
        B = pde.assemble(preconditioner)
        for bc in bcs:
            bc.apply(A)
            bc.apply(B)

        if compute_approximate_condition_number:
            print("Checking approximate condition number")
            csr_A = dolfin_to_scipy(A)
            csr_B = dolfin_to_scipy(B)
            print("   Computing maximum eigenvalues...", flush=True)
            max_eigs = spla.eigs(csr_A, M=csr_B, which='LM', return_eigenvectors=False, k=6, ncv=20, maxiter=csr_A.shape[0]*100)
            print("   Computing minimum eigenvalues...", flush=True)
            min_eigs = spla.eigs(csr_A, M=csr_B, sigma=0, return_eigenvectors=False, k=6, ncv=20, maxiter=csr_A.shape[0]*100)
            cond_approx = abs(max_eigs).max() / abs(min_eigs).min()
            print(f"   Condition number: {cond_approx:.2g}")
            condition_numbers['Condition number (approximate)'].append(cond_approx)

        if compute_condition_number:
            print("Checking condition number")
            np_A = A.array()
            np_B = B.array()
            print("   Computing eigenvalues...", flush=True)
            eigs = np.abs(sla.eigvals(np_A, np_B))
            cond_exact = eigs.max() / eigs.min()
            print(f"   Condition number: {cond_exact:.2g}")
            condition_numbers['Condition number (exact)'].append(cond_exact)

cond_df = pd.DataFrame(condition_numbers)
if compute_condition_number:
    print("Exact condition number")
    print(pd.pivot_table(cond_df, 'Condition number (exact)', 'Num timesteps', 'Spatial density'))

if compute_approximate_condition_number:
    print("Approximate condition number")
    print(pd.pivot_table(cond_df, 'Condition number (approximate)', 'Num timesteps', 'Spatial density'))

cond_df.to_csv("cond.csv")
