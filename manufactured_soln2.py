import sys
from tqdm import trange
from pathlib import Path
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import fenics as pde
import matplotlib.pyplot as plt
from ufl import nabla_div, dx, inner, nabla_grad

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


def fenics_contour(F, mesh_size=None, ax=None):
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

    vmax = np.max(np.abs(zz))
    vmin = -vmax
    ax.contourf(xx, yy, zz.reshape(xx.shape), vmin=-1, vmax=1, cmap="coolwarm", levels=100)
    ax.set_title(f"{zz.max():1f}, {zz.min():.1f}")
    ax.axis("equal")


class BiotSystem:
    def __init__(self, mesh, shear_modulus, lame_parameter, biot_coefficient, hydraulic_conductivity, force_field, source_field, dt):
        self.mesh = mesh
        self.shear_modulus = shear_modulus
        self.lame_parameter = lame_parameter
        self.biot_coefficient = biot_coefficient
        self.hydraulic_conductivity = hydraulic_conductivity
        self.force_field = force_field
        self.source_field = source_field
        self.dt = dt
        self.function_space = self._get_function_space()
        self.solution = pde.Function(self.function_space)
        self.previous_solution = pde.Function(self.function_space)
        self.trial_functions = pde.TrialFunctions(self.function_space)
        self.test_functions = pde.TestFunctions(self.function_space)

    def _get_function_space(self):
        P2 = pde.VectorElement("CG", self.mesh.ufl_cell(), 2)
        P1 = pde.FiniteElement("CG", self.mesh.ufl_cell(), 1)
        P2P1P1 = pde.MixedElement([P2, P1, P1])
        W = pde.FunctionSpace(self.mesh, P2P1P1)
        return W

    def get_lhs(self):
        mu = self.shear_modulus
        lambda_ = self.lame_parameter
        alpha = self.biot_coefficient
        kappa = self.hydraulic_conductivity
        dt = self.dt

        u, pT, pF = self.trial_functions
        v, qT, qF = self.test_functions

        a = (
              inner(2*mu*epsilon(u), epsilon(v)) - pT*nabla_div(v)
            - nabla_div(u)*qT                    - (1 / lambda_) * pT*qT + (alpha / lambda_)*pF*qT
                                                 + (alpha/lambda_)*pT*qF - 2*(alpha*alpha / lambda_)*pF*qF - dt*kappa*inner(nabla_grad(pF), nabla_grad(qF))
        )*dx
        return a

    def get_preconditioner(self):
        mu = self.shear_modulus
        lambda_ = self.lame_parameter
        alpha = self.biot_coefficient
        kappa = self.hydraulic_conductivity
        dt = self.dt
        u, pT, pF = self.trial_functions
        v, qT, qF = self.test_functions

        a = (
              + inner(2*mu*epsilon(u), epsilon(v)) 
              + pT*qT
              + 2*(alpha*alpha / lambda_)*pF*qF + inner(dt*kappa*nabla_grad(pF), nabla_grad(qF))
        )*dx
        return a

    def get_rhs(self):
        mu = self.shear_modulus
        lambda_ = self.lame_parameter
        alpha = self.biot_coefficient
        kappa = self.hydraulic_conductivity
        f = self.force_field
        g = self.source_field
        dt = self.dt
        u_, pT_, pF_ = pde.split(self.previous_solution)

        L = inner(f, v)*dx + dt*g*qF*dx
        timestep = qF * ((alpha / lambda_)*pT_ - 2*(alpha*alpha/lambda_)*pF_) * dx
        return L + timestep

#%%
compute_condition_number = True
compute_approximate_condition_number = True
run_simulation = False

#%% 
nT = int(sys.argv[1])
nX = int(sys.argv[2])
mesh = pde.UnitSquareMesh(nX, nX)
P2 = pde.VectorElement("CG", mesh.ufl_cell(), 2)
P1 = pde.FiniteElement("CG", mesh.ufl_cell(), 1)
P2P1P1 = pde.MixedElement([P2, P1, P1])
W = pde.FunctionSpace(mesh, P2P1P1)


#%% Set boundary conditions
u_dirichlet = "on_boundary" # "near(x[0], 1) || near(x[1], 0) || near(x[1], 1)"
u_neumann = f"!{u_dirichlet}" #f"near(x[0], 0) && on_boundary"
p_dirichlet = "on_boundary"
p_neumann = f"!{u_neumann}"

u_bc_mf = pde.MeshFunction('size_t', mesh, 1)
pde.CompiledSubDomain(u_neumann).mark(u_bc_mf, 1)
ds_force_region = pde.ds(subdomain_data=u_bc_mf)(1)
p_bc_mf = pde.MeshFunction('size_t', mesh, 1)
pde.CompiledSubDomain(p_neumann).mark(p_bc_mf, 1)
ds_velocity = pde.ds(subdomain_data=p_bc_mf)(1)


#%% Physical parameters
mu = pde.Constant(0.5)
lambda_ = pde.Constant(1)
alpha = pde.Constant(1)
kappa = pde.Constant(1)
dt = pde.Constant(1/nT)


#%% Define U and P
I = pde.Identity(mesh.geometric_dimension())
t = pde.Constant(0)
x = pde.SpatialCoordinate(mesh)
n = pde.FacetNormal(mesh)
zero = pde.Constant(0)
epsilon = pde.nabla_grad

U = pde.as_vector((
    pde.Constant(1)*pde.sin(x[0])*pde.sin(x[1])*pde.exp(t),
    pde.Constant(1)*pde.sin(x[0])*pde.sin(x[1])*pde.exp(t),
))
P = pde.Constant(1)*pde.sin(x[0])*pde.sin(x[1])*pde.exp(t)
U_dot = pde.as_vector((
    pde.Constant(1)*pde.sin(x[0])*pde.sin(x[1])*pde.exp(t),
    pde.Constant(1)*pde.sin(x[0])*pde.sin(x[1])*pde.exp(t)
))
P_dot = pde.Constant(1)*pde.sin(x[0])*pde.sin(x[1])*pde.exp(t)


P_total = -lambda_ * nabla_div(U) + P
P_total_dot = -lambda_ * nabla_div(U_dot) + P_dot

#%% Boundary stresses
displacement_stress = -2 * mu * epsilon(U) + P_total*I
fluid_velocity = kappa * nabla_grad(P)



#%% Setup manufactured solution
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

solution = system.solution #pde.Function(W)
previous_solution = system.previous_solution # pde.Function(W)
u, pT, pF = system.trial_functions
v, qT, qF = system.test_functions

a = system.get_lhs()
L = system.get_rhs() + dt*inner(pde.dot(fluid_velocity, n), qF)*ds_velocity
# + inner(displacement_stress*n, v)*ds_force_region

#%% Setup BCs
true_u = pde.project(U, rhs_V)
true_p_total = pde.project(P_total, rhs_Q)
true_p = pde.project(P, rhs_Q)
bcs = [
    pde.DirichletBC(W.sub(0), true_u, u_dirichlet),
    pde.DirichletBC(W.sub(2), true_p, p_dirichlet),
]


#%% Setup initial conditions
print("Setting initial conditions", flush=True)
a_initial = inner(u, v)*dx + inner(pT, qT)*dx + inner(pF, qF)*dx
L_initial = inner(true_u, v)*dx + inner(true_p_total, qT)*dx + inner(true_p, qF)*dx
pde.solve(a_initial == L_initial, solution)

print("Setting up solver")
A = pde.assemble(a)
B = pde.assemble(system.get_preconditioner())
for bc in bcs:
    bc.apply(A)
    bc.apply(B)
solver = pde.KrylovSolver('gmres', 'amg')
solver.set_operators(A, B)

if compute_condition_number:
    print("Checking condition number")
    np_A = A.array()
    np_B = B.array()
    print("   Computing eigenvalues...", flush=True)
    eigs = np.abs(sla.eigvals(np_A, np_B))
    cond = eigs.max() / eigs.min()
    print(f"   Condition number: {cond:.2g}")

if compute_approximate_condition_number:
    print("Checking condition number")
    csr_A = dolfin_to_scipy(A)
    csr_B = dolfin_to_scipy(B)
    print("   Computing maximum eigenvalues...", flush=True)
    max_eigs = spla.eigs(csr_A, M=csr_B, which='LM', return_eigenvectors=False)
    print("   Computing minimum eigenvalues...", flush=True)
    min_eigs = spla.eigs(csr_A, M=csr_B, sigma=0, return_eigenvectors=False)
    cond = abs(max_eigs).max() / abs(min_eigs).min()
    print(f"   Condition number: {cond:.2g}")

if not run_simulation:
    sys.exit(0)

print("Starting timestepping", flush=True)
for i in trange(nT):
    t.assign(t(0) + dt(0))
    previous_solution.assign(solution)
    
    true_u = pde.project(U, bc_V)
    true_p = pde.project(P, bc_Q)
    bcs = [
        pde.DirichletBC(W.sub(0), true_u, u_dirichlet),
        pde.DirichletBC(W.sub(2), true_p, p_dirichlet),
    ]

    b = pde.assemble(L)
    for bc in bcs:
        bc.apply(b)

    solver.solve(solution.vector(), b)

if plot_result:
    plt.subplot(221)
    plt.title("True u")
    fenics_quiver(true_u)
    plt.subplot(224)
    plt.title("Estimated u")
    fenics_quiver(solution.split()[0])
    plt.subplot(222)
    plt.title("True pF")
    fenics_contour(true_p_total)
    plt.subplot(224)
    plt.title("Estimated pF")
    fenics_contour(solution.split()[1])
    plt.show()


true_u = pde.project(U, rhs_V)
true_p_total = pde.project(P_total, rhs_Q)
true_p = pde.project(P, rhs_Q)


err_u = pde.errornorm(true_u, solution.split()[0], 'h1')
err_pT = pde.errornorm(true_p_total, solution.split()[1], 'l2')
err_pF =  pde.errornorm(true_p, solution.split()[2], 'l2')


if not Path("results.csv").is_file():
    with open("results.csv", "w") as f:
        f.write("nT,nX,u,pT,pF\n")

with open("results.csv", "r") as f:
    data = f.read()

with open("results.csv", "w") as f:
    f.write(f"{data}" + f"{nT},{nX},{err_u},{err_pT},{err_pF}\n")

print(data + f"{nT},{nX},{err_u},{err_pT},{err_pF}\n")
