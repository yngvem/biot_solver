from tqdm import trange
import fenics as pde
import matplotlib.pyplot as plt
import pandas as pd
from ufl import nabla_div, dx, inner, nabla_grad, ds, dot
from biot_system.utils import fenics_contour, fenics_quiver
from biot_system import BiotSystem, epsilon


plot_result = True
pd.set_option("max_columns", 20)


#%% Setup logs
errors = {}
errors['Num timesteps'] = []
errors['Spatial density'] = []
errors['Setup'] = []
errors['u (H1)'] = []
errors['u (L2)'] = []
errors['pT (H1)'] = []
errors['pT (L2)'] = []
errors['pF (H1)'] = []
errors['pF (L2)'] = []
errors['t'] = []
errors['u scale'] = []
errors['p scale'] = []
errors['u-Dirichlet'] = []
errors['u-Neumann'] = []
errors['p-Dirichlet'] = []
errors['p-Neumann'] = []


#%% Physical parameters
mu = pde.Constant(0.5)
lambda_ = pde.Constant(1)
alpha = pde.Constant(1)
kappa = pde.Constant(1)
zero = pde.Constant(0)
I = pde.Identity(2)


#%% Set boundary conditions
boundaries = {
    'left': pde.CompiledSubDomain("near(x[0], 0)"),
    'right': pde.CompiledSubDomain("near(x[0], 1)"),
    'top': pde.CompiledSubDomain("near(x[1], 1)"),
    'bottom': pde.CompiledSubDomain("near(x[1], 0)"),
}
boundary_indices = {
    'left': 1,
    'right': 2,
    'top': 3,
    'bottom': 4
}

all_bc_settings = {
    'dirichlet': {
        'u_dirichlet': ['left', 'right', 'top', 'bottom'],
        'u_neumann': [],
        'p_dirichlet': ['left', 'right', 'top', 'bottom'],
        'p_neumann': [],
    },
    'neumann': {
        'u_dirichlet': [],
        'u_neumann': ['left', 'right', 'top', 'bottom'],
        'p_dirichlet': [],
        'p_neumann': ['left', 'right', 'top', 'bottom'],
    },
    'u_dirichlet-p_neumann': {
        'u_dirichlet': ['left', 'right', 'top', 'bottom'],
        'u_neumann': [],
        'p_dirichlet': [],
        'p_neumann': ['left', 'right', 'top', 'bottom'],
    },
    'u_neumann-p_dirichlet': {
        'u_dirichlet': [],
        'u_neumann': ['left', 'right', 'top', 'bottom'],
        'p_dirichlet': ['left', 'right', 'top', 'bottom'],
        'p_neumann': [],
    },
    'u_mixed-p_dirichlet': {
        'u_dirichlet': ['left'],
        'u_neumann': ['right', 'top', 'bottom'],
        'p_dirichlet': ['left', 'right', 'top', 'bottom'],
        'p_neumann': [],
    },
    'u_dirichlet-p_mixed': {
        'u_dirichlet': ['left', 'right', 'top', 'bottom'],
        'u_neumann': [],
        'p_dirichlet': ['left'],
        'p_neumann': ['right', 'top', 'bottom'],
    },
    'u_mixed-p_neumann': {
        'u_dirichlet': ['left'],
        'u_neumann': ['right', 'top', 'bottom'],
        'p_dirichlet': [],
        'p_neumann': ['left', 'right', 'top', 'bottom'],
    },
    'u_neumann-p_mixed': {
        'u_dirichlet': [],
        'u_neumann': ['left', 'right', 'top', 'bottom'],
        'p_dirichlet': ['left'],
        'p_neumann': ['right', 'top', 'bottom'],
    },
    'u_mixed-p_mixed_opposite': {
        'u_dirichlet': ['right', 'top', 'bottom'],
        'u_neumann': ['left'],
        'p_dirichlet': ['left'],
        'p_neumann': ['right', 'top', 'bottom'],
    },
    'u_mixed-p_mixed': {
        'u_dirichlet': ['left'],
        'u_neumann': ['right', 'top', 'bottom'],
        'p_dirichlet': ['left'],
        'p_neumann': ['right', 'top', 'bottom'],
    },
}


for nT in [8]:
    print("\nnT", nT, flush=True)
    for nX in [4, 8, 16]:
        print("\nnX", nX, flush=True)
        for setup, bc_settings in all_bc_settings.items():
            u_dirichlet = bc_settings['u_dirichlet']
            p_dirichlet = bc_settings['p_dirichlet']
            u_neumann   = bc_settings['u_neumann']
            p_neumann   = bc_settings['p_neumann']
            print("Setup:", setup, "out of", len(all_bc_settings))


            #%% Start simulation
            for U_scale in [0, 1]:
                U_scale = pde.Constant(U_scale)
                for P_scale in [0, 1]:
                    P_scale = pde.Constant(P_scale)
                    dt = pde.Constant(1/nT)
                    t = pde.Constant(0)
                    mesh = pde.UnitSquareMesh(nX, nX)

                    #%% Setup boundary regions
                    mf = pde.MeshFunction('size_t', mesh, 1)
                    for name, boundary in boundaries.items():
                        boundary.mark(mf, boundary_indices[name])
                    pde.File("mf.pvd") << mf

                    #%% Define U and P
                    x = pde.SpatialCoordinate(mesh)
                    n = pde.FacetNormal(mesh)

                    U = pde.as_vector((
                        U_scale*pde.cos(x[0])*pde.cos(x[1]),  #*t,
                        U_scale*pde.cos(x[0])*pde.cos(x[1])  #*t
                    ))
                    P = P_scale*pde.cos(x[0])*pde.cos(x[1])  #*t
                    U_dot = pde.as_vector((
                        zero*U_scale*pde.cos(x[0])*pde.cos(x[1]),
                        zero*U_scale*pde.cos(x[0])*pde.cos(x[1])
                    ))
                    P_dot = zero*P_scale*pde.cos(x[0])*pde.cos(x[1])

                    P_total = -lambda_ * nabla_div(U) + P
                    P_total_dot = -lambda_ * nabla_div(U_dot) + P_dot

                    displacement_stress = 2 * mu * epsilon(U) - P_total*I
                    fluid_velocity = kappa*nabla_grad(P_total)

                    #%% Setup RHS
                    rhs_V = pde.VectorFunctionSpace(mesh, 'CG', 5)
                    rhs_Q = pde.FunctionSpace(mesh, 'CG', 5)
                    bc_V = pde.VectorFunctionSpace(mesh, 'CG', 2)
                    bc_Q = pde.FunctionSpace(mesh, 'CG', 1)

                    f = -nabla_div(2*mu*epsilon(U)) + nabla_grad(P_total)
                    g = (alpha/lambda_)*P_total_dot - 2*(alpha*alpha/lambda_)*P_dot + kappa*nabla_div(nabla_grad(P))

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

                        errors['Num timesteps'].append(nT)
                        errors['Spatial density'].append(nX)
                        errors['t'].append(t(0))
                        errors['u scale'].append(U_scale(0))
                        errors['p scale'].append(P_scale(0))
                        errors['u-Dirichlet'].append(u_dirichlet)
                        errors['u-Neumann'].append(u_neumann)
                        errors['p-Dirichlet'].append(p_dirichlet)
                        errors['p-Neumann'].append(p_neumann)
                        errors['Setup'].append(setup)
                        errors['u (H1)'].append(pde.errornorm(true_u, solution.split()[0], 'h1'))
                        errors['u (L2)'].append(pde.errornorm(true_u, solution.split()[0], 'l2'))
                        errors['pT (H1)'].append(pde.errornorm(true_p_total, solution.split()[1], 'h1'))
                        errors['pT (L2)'].append(pde.errornorm(true_p_total, solution.split()[1], 'l2'))
                        errors['pF (H1)'].append(pde.errornorm(true_p, solution.split()[2], 'h1'))
                        errors['pF (L2)'].append(pde.errornorm(true_p, solution.split()[2], 'l2'))
                        pd.DataFrame(errors).to_csv("convergence/errors.csv")

                        if plot_result:
                            fig = plt.figure()
                            plt.subplot(321)
                            plt.title("True u")
                            fenics_quiver(true_u)
                            plt.subplot(323)
                            plt.title("Estimated u")
                            fenics_quiver(solution.split()[0])
                            plt.subplot(325)
                            plt.title("Error u")
                            fenics_quiver(pde.project(true_u - solution.split()[0], bc_V))

                            plt.subplot(322)
                            plt.title("True pF")
                            fenics_contour(true_p)
                            plt.subplot(324)
                            plt.title("Estimated pF")
                            fenics_contour(solution.split()[2])
                            plt.subplot(326)
                            plt.title("Error pF")
                            fenics_contour(pde.project(true_p - solution.split()[2], bc_Q))
                            plt.savefig(f"convergence_static/figs/bcs_{setup}--u_{U_scale(0)}-p_{P_scale(0)}--nT_{nT}-nX_{nX}--ti_{i}.png", dpi=200)
                            plt.close(fig)

