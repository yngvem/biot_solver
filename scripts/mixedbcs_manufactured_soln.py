from tqdm import trange
import fenics as pde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ufl import nabla_div, dx, inner, nabla_grad
from biot_system.utils import fenics_contour, fenics_quiver
from biot_system import BiotSystem, epsilon


plot_result = True
pd.set_option("max_columns", 20)

#%% Set boundary conditions
u_dirichlet = "near(x[0], 1) || near(x[1], 0) || near(x[1], 1)"
u_neumann = "near(x[0], 0)"
p_dirichlet = "on_boundary"
p_neumann = "false"

#%% Physical parameters
mu = pde.Constant(0.5)
lambda_ = pde.Constant(1)
alpha = pde.Constant(1)
kappa = pde.Constant(1)
zero = pde.Constant(0)
I = pde.Identity(2)

# Setup error dict
errors = {}
errors['Num timesteps'] = []
errors['Spatial density'] = []
errors['u (H1)'] = []
errors['u (L2)'] = []
errors['pT (H1)'] = []
errors['pT (L2)'] = []
errors['pF (H1)'] = []
errors['pF (L2)'] = []

#%% Start simulation
for nT in [8]:
    print("\nnT", nT, flush=True)
    for nX in [4, 8, 16]:
        print("\nnX", nX, flush=True)
        dt = pde.Constant(1/nT)
        t = pde.Constant(0)
        mesh = pde.UnitSquareMesh(nX, nX)

        #%% Setup boundary regions
        u_bc_mf = pde.MeshFunction('size_t', mesh, 1)
        p_bc_mf = pde.MeshFunction('size_t', mesh, 1)
        pde.CompiledSubDomain(u_neumann).mark(u_bc_mf, 1)
        pde.CompiledSubDomain(p_neumann).mark(p_bc_mf, 1)
        ds_force_region = pde.ds(subdomain_data=u_bc_mf)(1)
        ds_velocity = pde.ds(subdomain_data=p_bc_mf)(1)

        #%% Define U and P
        x = pde.SpatialCoordinate(mesh)
        n = pde.FacetNormal(mesh)

        U = pde.as_vector((
            pde.sin(x[0])*pde.sin(x[1])*t, #pde.exp(t),
            pde.sin(x[0])*pde.sin(x[1])*t #pde.exp(t),
        ))
        P = pde.sin(x[0])*pde.sin(x[1])*t #pde.exp(t)
        U_dot = pde.as_vector((
            pde.sin(x[0])*pde.sin(x[1]),#*pde.exp(t),
            pde.sin(x[0])*pde.sin(x[1])#*pde.exp(t)
        ))
        P_dot = pde.sin(x[0])*pde.sin(x[1])#*pde.exp(t)

        P_total = -lambda_ * nabla_div(U) + P
        P_total_dot = -lambda_ * nabla_div(U_dot) + P_dot

        displacement_stress = -2 * mu * epsilon(U) + P_total*I

        #%% Setup RHS
        rhs_V = pde.VectorFunctionSpace(mesh, 'CG', 5)
        rhs_Q = pde.FunctionSpace(mesh, 'CG', 5)
        bc_V = pde.VectorFunctionSpace(mesh, 'CG', 2)
        bc_Q = pde.FunctionSpace(mesh, 'CG', 1)

        f = -nabla_div(2*mu*epsilon(U)) + nabla_grad(P_total)
        g = (alpha/lambda_)*P_total_dot - 2*(alpha*alpha/lambda_)*P_dot + kappa*nabla_div(nabla_grad(P))

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
        L = system.get_rhs() - inner(displacement_stress*n, v)*ds_force_region  # + dt*inner(pde.dot(fluid_velocity, n), qF)*ds_velocity

        #%% Setup BCs
        true_u = pde.project(U, rhs_V)
        true_p_total = pde.project(P_total, rhs_Q)
        true_p = pde.project(P, rhs_Q)
        bcs = [
            pde.DirichletBC(system.function_space.sub(0), true_u, u_dirichlet),
            pde.DirichletBC(system.function_space.sub(2), true_p, p_dirichlet),
        ]

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

        print("Starting timestepping", flush=True)
        for i in trange(nT):
            t.assign(t(0) + dt(0))
            previous_solution.assign(solution)

            true_u = pde.project(U, bc_V)
            true_p = pde.project(P, bc_Q)
            bcs = [
                pde.DirichletBC(system.function_space.sub(0), true_u, u_dirichlet),
                pde.DirichletBC(system.function_space.sub(2), true_p, p_dirichlet),
            ]

            b = pde.assemble(L)
            for bc in bcs:
                bc.apply(b)

            solver.solve(solution.vector(), b)

        if plot_result:
            plt.subplot(221)
            plt.title("True u")
            fenics_quiver(true_u)
            plt.subplot(223)
            plt.title("Estimated u")
            fenics_quiver(solution.split()[0])
            plt.subplot(222)
            plt.title("True pF")
            fenics_contour(true_p)
            plt.subplot(224)
            plt.title("Estimated pF")
            fenics_contour(solution.split()[2])
            plt.show()

        true_u = pde.project(U, rhs_V)
        true_p_total = pde.project(P_total, rhs_Q)
        true_p = pde.project(P, rhs_Q)
        print()
        print("------")
        print("p(0, 1)", true_p(0, 1), solution.split()[2](0, 1), flush=True)
        print("p(0, 0.3)", true_p(0, 0.3), solution.split()[2](0, 0.3), flush=True)
        print("p(1, 1)", true_p(1, 1), solution.split()[2](1, 1), flush=True)
        print("u(0, 1)", true_u(0, 1), solution.split()[0](0, 1), flush=True)
        print("u(0, 0.3)", true_u(0, 0.3), solution.split()[0](0, 0.3), flush=True)
        print("u(1, 1)", true_u(1, 1), solution.split()[0](1, 1), flush=True)

        errors['Num timesteps'].append(nT)
        errors['Spatial density'].append(nX)
        errors['u (H1)'].append(pde.errornorm(true_u, solution.split()[0], 'h1'))
        errors['u (L2)'].append(pde.errornorm(true_u, solution.split()[0], 'l2'))
        errors['pT (H1)'].append(pde.errornorm(true_p_total, solution.split()[1], 'h1'))
        errors['pT (L2)'].append(pde.errornorm(true_p_total, solution.split()[1], 'l2'))
        errors['pF (H1)'].append(pde.errornorm(true_p, solution.split()[2], 'h1'))
        errors['pF (L2)'].append(pde.errornorm(true_p, solution.split()[2], 'l2'))

        errors_df = pd.DataFrame(errors)
        errors_df.to_csv("result_mixed_bcs.csv")
        print()
        print(errors_df)
        if len(errors_df) > 1:
            print(np.log(errors_df).diff())
