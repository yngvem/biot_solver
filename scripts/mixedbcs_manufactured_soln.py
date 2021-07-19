import itertools
from pathlib import Path

import fenics as pde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange
from ufl import dot, ds, dx, inner, nabla_div, nabla_grad

from biot_system import TotalPressureBiotSystem, epsilon
from biot_system.benchmarks import static_P, static_U
from biot_system.utils import plot_solution

plot_result = True


def run_simulation(P_scale, U_scale, nT, nX, mu_value, lambda_value, kappa_value, alpha_value, u_dirichlet, p_dirichlet):
    mu = pde.Constant(mu_value)
    lambda_ = pde.Constant(lambda_value)
    kappa = pde.Constant(kappa_value)
    P_scale = pde.Constant(P_scale)
    U_scale = pde.Constant(U_scale)
    dt = pde.Constant(1/nT)
    t = pde.Constant(0)
    mesh = pde.UnitSquareMesh(nX, nX)

    #%% Setup boundary regions
    mf = pde.MeshFunction('size_t', mesh, 1)
    for name, boundary in boundaries.items():
        boundary.mark(mf, boundary_indices[name])

    #%% Define U and P
    x = pde.SpatialCoordinate(mesh)
    n = pde.FacetNormal(mesh)

    U, U_dot = static_U(x, t, scale=U_scale)
    P, P_dot = static_P(x, t, scale=P_scale)
    P_total = -lambda_ * nabla_div(U) + P
    P_total_dot = -lambda_ * nabla_div(U_dot) + P_dot

    displacement_stress = 2 * mu * epsilon(U) - P_total*I
    fluid_velocity = kappa*nabla_grad(P_total)

    f = -nabla_div(2*mu*epsilon(U)) + nabla_grad(P_total)
    g = (alpha/lambda_)*P_total_dot - 2*(alpha*alpha/lambda_)*P_dot + kappa*nabla_div(nabla_grad(P))

    #%% Setup RHS
    rhs_V = pde.VectorFunctionSpace(mesh, 'CG', 5)
    rhs_Q = pde.FunctionSpace(mesh, 'CG', 5)
    bc_V = pde.VectorFunctionSpace(mesh, 'CG', 2)
    bc_Q = pde.FunctionSpace(mesh, 'CG', 1)

    #%% Setup bilinear form
    system = TotalPressureBiotSystem(
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
    L = system.get_rhs() + inner(displacement_stress*n, v)*ds + dt*inner(dot(fluid_velocity, n), qF)*ds

    #%% Setup BCs
    true_u = pde.project(U, rhs_V)
    true_p_total = pde.project(P_total, rhs_Q)
    true_p = pde.project(P, rhs_Q)
    bcs = []
    bcs += [pde.DirichletBC(system.function_space.sub(0), true_u, mf, boundary_indices[name]) for name in u_dirichlet]
    bcs += [pde.DirichletBC(system.function_space.sub(2), true_p, mf, boundary_indices[name]) for name in p_dirichlet]

    #%% Setup initial conditions
    print("Setting initial conditions", flush=True)
    a_initial = inner(u, v)*dx + inner(pT, qT)*dx + inner(pF, qF)*dx
    L_initial = inner(true_u, v)*dx + inner(true_p_total, qT)*dx + inner(true_p, qF)*dx
    pde.solve(a_initial == L_initial, solution)

    print("Setting up solver")
    A = pde.assemble(a)
    for bc in bcs:
        bc.apply(A)
    solver = pde.LUSolver()
    solver.set_operator(A)
    print(pde.errornorm(true_u, solution.split()[0]))

    print("Starting timestepping", flush=True)
    if plot_result:
        fig, axes = plot_solution(true_u, true_p, solution.split()[0], solution.split()[2], bc_V, bc_Q)
        fig.savefig(fig_path / f"mu_{mu_value}-lambda_{lambda_value}-kappa_{kappa_value}-alpha_{alpha_value}-dirichlet_u_{u_dirichlet}-dirichlet_p_{p_dirichlet}--u_{U_scale(0)}-p_{P_scale(0)}--nT_{nT:02d}-nX_{nX:02d}--ti_00.png", dpi=200)
        plt.close(fig)

    for i in trange(nT):
        t.assign(t(0) + dt(0))
        previous_solution.assign(solution)

        true_u = pde.project(U, bc_V)
        true_p = pde.project(P, bc_Q)
        bcs = []
        bcs += [pde.DirichletBC(system.function_space.sub(0), true_u, mf, boundary_indices[name]) for name in u_dirichlet]
        bcs += [pde.DirichletBC(system.function_space.sub(2), true_p, mf, boundary_indices[name]) for name in p_dirichlet]

        b = pde.assemble(L)
        for bc in bcs:
            bc.apply(b)

        solver.solve(solution.vector(), b)

    true_u = pde.project(U, rhs_V)
    true_p_total = pde.project(P_total, rhs_Q)
    true_p = pde.project(P, rhs_Q)
    bc_u = pde.project(U, bc_V)
    bc_p_total = pde.project(P_total, bc_Q)
    bc_p = pde.project(P, bc_Q)
    est_u = pde.project(solution.split()[0], bc_V)
    est_p_total = pde.project(solution.split()[1], bc_Q)
    est_p = pde.project(solution.split()[2], bc_Q)

    results = {}
    results['Num timesteps'] = (nT)
    results['Spatial density'] = (nX)
    results['mu'] = (mu_value)
    results['lambda'] = (lambda_value)
    results['kappa'] = (kappa_value)
    results['alpha'] = (alpha_value)
    results['t'] = (t(0))
    results['u scale'] = (U_scale(0))
    results['p scale'] = (P_scale(0))
    results['u-Dirichlet'] = (u_dirichlet)
    results['p-Dirichlet'] = (p_dirichlet)
    results['Setup'] = (setup)
    results['u (H1)'] = (pde.errornorm(true_u, solution.split()[0], 'h1'))
    results['u (L2)'] = (pde.errornorm(true_u, solution.split()[0], 'l2'))
    results['u (Linf)'] = (np.max(np.abs(est_u.vector() - bc_u.vector())))
    results['pT (H1)'] = (pde.errornorm(true_p_total, solution.split()[1], 'h1'))
    results['pT (L2)'] = (pde.errornorm(true_p_total, solution.split()[1], 'l2'))
    results['pT (Linf)'] = (np.max(np.abs(est_p_total.vector() - bc_p_total.vector())))
    results['pF (H1)'] = (pde.errornorm(true_p, solution.split()[2], 'h1'))
    results['pF (L2)'] = (pde.errornorm(true_p, solution.split()[2], 'l2'))
    results['pF (Linf)'] = (np.max(np.abs(est_p.vector() - bc_p.vector())))


    if plot_result:
        fig, axes = plot_solution(true_u, true_p, solution.split()[0], solution.split()[2], bc_V, bc_Q)
        fig.savefig(fig_path / f"mu_{mu_value}-lambda_{lambda_value}-kappa_{kappa_value}-alpha_{alpha_value}-dirichlet_u_{u_dirichlet}-dirichlet_p_{p_dirichlet}--u_{U_scale(0)}-p_{P_scale(0)}--nT_{nT:02d}-nX_{nX:02d}--ti_{i+1:02d}.png", dpi=200)
        plt.close(fig)
    return results

#%% Setup logs
errors = {}
errors['Num timesteps'] = []
errors['Spatial density'] = []
errors['Setup'] = []
errors['u (H1)'] = []
errors['u (L2)'] = []
errors['u (Linf)'] = []
errors['pT (H1)'] = []
errors['pT (L2)'] = []
errors['pT (Linf)'] = []
errors['pF (H1)'] = []
errors['pF (L2)'] = []
errors['pF (Linf)'] = []
errors['t'] = []
errors['u scale'] = []
errors['p scale'] = []
errors['u-Dirichlet'] = []
errors['p-Dirichlet'] = []
errors['mu'] = []
errors['lambda'] = []
errors['kappa'] = []
errors['alpha'] = []


#%% General constants
zero = pde.Constant(0)
I = pde.Identity(2)


#%% Set boundary conditions
boundaries = {
    'l': pde.CompiledSubDomain("near(x[0], 0)"),
    'r': pde.CompiledSubDomain("near(x[0], 1)"),
    't': pde.CompiledSubDomain("near(x[1], 1)"),
    'b': pde.CompiledSubDomain("near(x[1], 0)"),
}
boundary_indices = {
    'l': 1,
    'r': 2,
    't': 3,
    'b': 4
}

all_bc_settings = {
    'u_dirichlet-p_neumann': {
        'u_dirichlet': 'lrtb',
        'p_dirichlet': '',
    },
    'u_neumann-p_dirichlet': {
        'u_dirichlet': '',
        'p_dirichlet': 'lrtb',
    },
    'u_mixed-p_dirichlet': {
        'u_dirichlet': 'l',
        'p_dirichlet': 'lrtb',
    },
    'u_dirichlet-p_mixed': {
        'u_dirichlet': 'lrtb',
        'p_dirichlet': 'l',
    },
    'u_mixed-p_neumann': {
        'u_dirichlet': 'l',
        'p_dirichlet': '',
    },
    'u_neumann-p_mixed': {
        'u_dirichlet': '',
        'p_dirichlet': 'l',
    },
    'u_mixed-p_mixed_opposite': {
        'u_dirichlet': 'rtb',
        'p_dirichlet': 'l',
    },
    'u_mixed-p_mixed': {
        'u_dirichlet': 'l',
        'p_dirichlet': 'l',
    },
}

output_path = Path("convergence_gridsearch_static")
fig_path = output_path / "fig"
fig_path.mkdir(parents=True, exist_ok=True)
experiment_id = 0


#%% Physical parameters
alpha_value = 1
alpha = pde.Constant(alpha_value)



values = [0, 1e-10, 1]
for mu_value, lambda_value, kappa_value in itertools.product(values, repeat=3):

        setup = "u_mixed-p_mixed_opposite"
        bc_settings = all_bc_settings[setup]
        print("Setup:", setup, "out of", len(all_bc_settings))
        for nT in [1, 8]:
            print("\nnT", nT, flush=True)
            for nX in [8, 16]:
                print("\nnX", nX, flush=True)
                for U_scale in [0, 1]:
                    for P_scale in [0, 1]:
                        u_dirichlet = bc_settings['u_dirichlet']
                        p_dirichlet = bc_settings['p_dirichlet']

                        if P_scale == U_scale == 0:
                            continue

                        results = run_simulation(P_scale, U_scale, nT, nX, mu_value, lambda_value, kappa_value, alpha_value, u_dirichlet, p_dirichlet)
                        for key, value in results.items():
                            errors[key].append(value)
                        pd.DataFrame(errors).to_csv(output_path / "errors.csv")
