import fenics as pde
from ufl import dx, inner, nabla_div, nabla_grad


def epsilon(u):
    return nabla_grad(u)


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
        v, qT, qF = self.test_functions
        u_, pT_, pF_ = pde.split(self.previous_solution)

        L = inner(f, v)*dx + dt*g*qF*dx
        timestep = qF * ((alpha / lambda_)*pT_ - 2*(alpha*alpha/lambda_)*pF_) * dx
        return L + timestep

