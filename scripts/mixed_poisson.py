from tqdm import trange
import fenics as pde
import matplotlib.pyplot as plt
import pandas as pd
from ufl import nabla_div, dx, inner
from biot_system.utils import fenics_contour
from biot_system import epsilon


plot_result = True


#%% Set boundary conditions
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

#%% Physical parameters
mu = pde.Constant(0.5)
lambda_ = pde.Constant(1)
alpha = pde.Constant(1)
kappa = pde.Constant(1)
zero = pde.Constant(0)

# Setup error dict
errors = {}
errors['Num timesteps'] = []
errors['Spatial density'] = []
errors['u (H1)'] = []
errors['u (L2)'] = []

#%% Start simulation
for nT in [8, 64]:
    print("\nnT", nT, flush=True)
    for nX in [4, 8, 16, 32]:
        print("\nnX", nX, flush=True)
        dt = pde.Constant(1/nT)
        t = pde.Constant(0)
        mesh = pde.UnitSquareMesh(nX, nX)


        # Mark boundaries
        mf = pde.MeshFunction('size_t', mesh, 1)
        for name, boundary in boundaries.items():
            boundary.mark(mf, boundary_indices[name])
        

        # Specify the type of the different boundaries
        neumann = ['top', 'left', 'bottom']
        dirichlet = ['right']
        #neumann = []
        #dirichlet = ['top', 'left', 'bottom', 'right']

        ds = pde.ds

        #%% Define U and P
        x = pde.SpatialCoordinate(mesh)
        n = pde.FacetNormal(mesh)

        U = pde.sin(x[0])*pde.sin(x[1])*t#*pde.exp(t)
        U_dot = pde.sin(x[0])*pde.sin(x[1])#*pde.exp(t)


        #%% Setup RHS
        rhs_V = pde.FunctionSpace(mesh, 'CG', 5)
        bc_V = pde.FunctionSpace(mesh, 'CG', 2)

        f = U_dot - nabla_div(epsilon(U))


        #%% Setup bilinear form
        solution = pde.Function(bc_V)
        previous_solution = pde.Function(bc_V)
        u = pde.TrialFunction(bc_V)
        v = pde.TestFunction(bc_V)
        a = inner(u, v)*dx + dt*inner(epsilon(u), epsilon(v))*dx
        L = inner(solution, v)*dx + dt*inner(f, v)*dx + dt*inner(pde.dot(epsilon(U), n), v)*ds

        #%% Setup BCs
        true_u = pde.project(U, rhs_V)
        bcs = [
            pde.DirichletBC(bc_V, true_u, mf, boundary_indices[dirichlet[0]]),
        ]
        for name in dirichlet[1:]:
            bcs += [pde.DirichletBC(bc_V, true_u, mf, boundary_indices[name])]

        #%% Setup initial conditions
        print("Setting initial conditions", flush=True)
        solution.vector()[:] = pde.project(U, bc_V).vector()

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
            bcs = [
                pde.DirichletBC(bc_V, true_u, mf, boundary_indices[name]) for name in dirichlet
            ]

            b = pde.assemble(L)
            for bc in bcs:
                bc.apply(b)

            solver.solve(solution.vector(), b)

        if plot_result:
            plt.subplot(131)
            fenics_contour(true_u)
            plt.subplot(132)
            fenics_contour(solution)
            plt.subplot(133)
            fenics_contour(pde.project(true_u - solution, bc_V))
            plt.show()

        true_u = pde.project(U, rhs_V)

        errors['Num timesteps'].append(nT)
        errors['Spatial density'].append(nX)
        errors['u (H1)'].append(pde.errornorm(true_u, solution, 'h1'))
        errors['u (L2)'].append(pde.errornorm(true_u, solution, 'l2'))
        print(errors)

        pd.DataFrame(errors).to_csv("poisson_result.csv")
