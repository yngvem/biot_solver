import fenics as pde

zero = pde.Constant(0)


def static_U(x, t, scale):
    U = pde.as_vector((
        scale*pde.cos(x[0])*pde.cos(x[1]),
        scale*pde.cos(x[0])*pde.cos(x[1])
    ))
    U_dot = pde.as_vector((
        zero*x[0],
        zero*x[0]
    ))

    return U, U_dot


def static_P(x, t, scale):
    P = scale*pde.cos(x[0])*pde.cos(x[1])
    P_dot = zero*x[0]
    return P, P_dot


def linear_time_U(x, t, scale):
    U = pde.as_vector((
        scale*pde.cos(x[0])*pde.cos(x[1])*t,
        scale*pde.cos(x[0])*pde.cos(x[1])*t
    ))
    U_dot = pde.as_vector((
        scale*pde.cos(x[0])*pde.cos(x[1]),
        scale*pde.cos(x[0])*pde.cos(x[1])
    ))

    return U, U_dot


def linear_time_P(x, t, scale):
    scale = pde.Constant(scale)
    P = scale*pde.cos(x[0])*pde.cos(x[1])*t
    P_dot = scale*pde.cos(x[0])*pde.cos(x[1])
    return P, P_dot


def exponential_time_U(x, t, scale):
    U = pde.as_vector((
        scale*pde.cos(x[0])*pde.cos(x[1])*pde.exp(t),
        scale*pde.cos(x[0])*pde.cos(x[1])*pde.exp(t)
    ))
    U_dot = pde.as_vector((
        scale*pde.cos(x[0])*pde.cos(x[1])*pde.exp(t),
        scale*pde.cos(x[0])*pde.cos(x[1])*pde.exp(t)
    ))

    return U, U_dot


def exponential_time_P(x, t, scale):
    scale = pde.Constant(scale)
    P = scale*pde.cos(x[0])*pde.cos(x[1])*pde.exp(t)
    P_dot = scale*pde.cos(x[0])*pde.cos(x[1])*pde.exp(t)
    return P, P_dot
