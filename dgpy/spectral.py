import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.special


def leggausslobatto(N):
    return [
        (np.array([-1, 1]), np.array([1.0, 1.0])),
        (np.array([-1, 0, 1]), np.array([1.0, 4.0, 1.0]) / 3),
        (np.array([-1, -0.4472135954999579, 0.4472135954999579, 1]),
         np.array([1.0, 5.0, 5.0, 1.0]) / 6),
        (np.array([-1.0, -0.6546536707079773, 0.0, 0.6546536707079773, 1.0]),
         np.array([9.0, 49.0, 64.0, 49.0, 9.0]) / 90),
        (np.array([-1.0, -0.7650553239294646, -0.2852315164806452, 0.2852315164806452, 0.7650553239294646, 1.0]),
         np.array([1.0 / 15.0, 0.378474956297847, 0.554858377035486, 0.554858377035486, 0.378474956297847, 1.0 / 15.0])),
        (np.array([-1.0, -0.8302238962785669, -0.4688487934707142, 0.0,
                   0.4688487934707142, 0.8302238962785669, 1.0]),
         np.array([1.0 / 21.0, 0.276826047361566, 0.431745381209860,
                   0.487619047619048, 0.431745381209862, 0.276826047361567,
                   1.0 / 21.0])),
        (np.array([-1.0, -0.8717401485096066, -0.5917001814331421,
                   -0.2092992179024791, 0.2092992179024791, 0.5917001814331421,
                   0.8717401485096066, 1.0]),
         np.array([1.0 / 28.0, 0.210704227143507, 0.341122692483504,
                   0.412458794658705, 0.412458794658705, 0.341122692483504,
                   0.210704227143507, 1.0 / 28.0])),
        (np.array([-1.0, -0.8997579954114600, -0.6771862795107377,
                   -0.3631174638261783, 0.0, 0.3631174638261783,
                   0.6771862795107377, 0.8997579954114600, 1.0]),
         np.array([1.0 / 36.0, 0.165495361560806, 0.274538712500161,
                   0.346428510973042, 0.371519274376423, 0.346428510973042,
                   0.274538712500161, 0.165495361560806, 1.0 / 36.0])),
        (np.array([-1.0, -0.9195339081664589, -0.7387738651055048,
                   -0.4779249498104444, -0.1652789576663869, 0.1652789576663869,
                   0.4779249498104444, 0.7387738651055048, 0.9195339081664589,
                   1.0]),
         np.array([1.0 / 45.0, 0.133305990851069, 0.224889342063126,
                   0.292042683679679, 0.327539761183898, 0.327539761183898,
                   0.292042683679680, 0.224889342063126, 0.133305990851071,
                   1.0 / 45.0]))
    ][N - 2]


def lgl_points(N):
    """Returns N Legendre-Gauss-Lobatto collocation points."""
    return leggausslobatto(N)[0]


def lgl_weights(N):
    """Returns N Legendre-Gauss-Lobatto quadrature weights."""
    return leggausslobatto(N)[1]


def lg_points(N):
    """Returns N Legendre-Gauss collocation points."""
    return leggauss(N)[0]


def lg_weights(N):
    """Returns N Legendre-Gauss quadrature weights."""
    return leggauss(N)[1]


def logical_coords(x, bounds):
    """Maps inertial to logical coordinates for rectilinear domains

    Args:
      x: Inertial coordinates. The first dimension of the array must correspond
        to the dimension of the domain, i.e. x[0] is the x-coordinate, x[1] is
        the y-coordinate etc.
      bounds: The (lower, upper) bounds of the element in inertial coordinates,
        in every dimension. For example: [(0, 1), (0.5, 3.5)]
    """
    x = np.asarray(x, dtype=np.float)
    dim = len(x)
    bounds = np.asarray(bounds, dtype=np.float)
    assert bounds.shape == (dim, 2), (
        f"The 'bounds' must have shape ({dim}, 2) in {dim} dimensions, but "
        f"the shape is: {bounds.shape}")
    rs = np.ones(x.ndim, int)
    rs[0] = -1
    a = np.squeeze(np.diff(bounds)) / 2
    b = np.sum(bounds, axis=-1) / 2
    return (x - b.reshape(rs)) / a.reshape(rs)


def inertial_coords(xi, bounds):
    """Maps logical to inertial coordinates for rectilinear domains

    Args:
      x: Logical coordinates. The first dimension of the array must correspond
        to the dimension of the domain, i.e. xi[0] is the xi-coordinate, xi[1]
        is the eta-coordinate etc.
      bounds: The (lower, upper) bounds of the element in inertial coordinates,
        in every dimension. For example: [(0, 1), (0.5, 3.5)]
    """
    xi = np.asarray(xi, dtype=np.float)
    dim = len(xi)
    bounds = np.asarray(bounds, dtype=np.float)
    assert bounds.shape == (dim, 2), (
        f"The 'bounds' must have shape ({dim}, 2) in {dim} dimensions, but "
        f"the shape is: {bounds.shape}")
    x = np.zeros(xi.shape)
    for d in range(dim):
        x[d] = xi[d] * (bounds[d][1] - bounds[d][0]) / 2 + (bounds[d][1] + bounds[d][0]) / 2
    return x


# Functions copied from Harald Pfeiffer's lecture notes

def vandermonde_matrix(r):
    alpha = 0
    beta = 0
    N = len(r) - 1

    # Vandermonde matrix for Legendre polynomials
    # V[i,j] = P_j(r_i),  j=0,...,N,  i=0,...,len(r)-1
    V = np.zeros((len(r), N+1))
    for j in range(N+1):
        # scipy normalization determined by trial and error.
        # For **LAGRANGE POLY** ONLY, not general alpha, beta.
        # This makes the returned polynomials orthonormal
        normalization = np.sqrt((1.+2.*j)/2.)
        V[:, j] = scipy.special.eval_jacobi(j, alpha, beta, r)*normalization
        # or V[:,j] = scipy.special.legendre(j)(r)

        # check normalization
        # tmp_r, tmp_w = scipy.special.roots_jacobi(j+1, alpha, beta)
        # tmp_L=scipy.special.eval_jacobi(j, alpha, beta, tmp_r)*normalization
        # L_dot_L = sum(tmp_w*tmp_L*tmp_L)
        # print("j={}, (L,L)={}".format(j, L_dot_L))
    return V


def logical_differentiation_matrix(r):
    """Returns the differentiation matrix for the collocation points

    Args:
      r: Collocation points
    """
    V = vandermonde_matrix(r)
    Vinv = np.linalg.inv(V)

    alpha = 0
    beta = 0
    N = len(r) - 1

    # derivatives of Legendre polynomials, evaluated at quadrature points
    # Vr[i,j] = dP_j/dr(r_i),  j=0,...,N,  i=0,...,len(r)-1
    #   use dP_j/dr = sqrt(j(j+1)) J^{alpha+1,beta+1}_{j-1}  (H+W, Eq A2)
    #
    Vr = np.zeros((len(r), N+1))
    for j in range(1, N+1):
        # scipy normalization determined by trial and error.
        # For **LAGRANGE POLY** ONLY, not general alpha, beta.
        # This makes the returned polynomials orthonormal, conforming
        # to H+W conventions
        scipy_normalization = np.sqrt((1.+2.*j)*(j+1.)/(8.*j))
        normed_J = scipy.special.jacobi(
            j-1, alpha+1, beta+1)(r)*scipy_normalization
        Vr[:, j] = np.sqrt(j*(j+alpha+beta+1.))*normed_J  # H+W Eq. A2

        # - check normalization
        # - integrate by Legendre quadrature, to explicitly show weight-function in orthogonality
        # tmp_r, tmp_w = scipy.special.roots_jacobi(j+4, alpha, beta)
        # tmp_L=scipy.special.eval_jacobi(j-1, alpha+1, beta+1, tmp_r)*scipy_normalization
        # - evaluate orthogonality; note weight function (1-r)(1+r)
        # L_dot_L = sum(tmp_w*tmp_L*tmp_L*(1-tmp_r)*(1+tmp_r))
        # print("j={}, (L,L)={}".format(j, L_dot_L))

    # derivatives of Lagrange interpolating polynomials
    #    Dr(i,j) = dl_j/dr(r=r_i),
    # where  l_j(r_i) = delta_{ij}
    # compute using P_j(r) = V[i,j]*l_i(r) =>  V[i,j] dl_i/dr = dP_j/dr     (*)
    #     => V^{-T} V^T[j,i] dl_i/dr = V^{-T} dP_j/dr
    Dr = np.matmul(Vr, Vinv)
    return Dr


def logical_mass_matrix(r):
    """Returns the mass matrix for the collocation points.

    Args:
      r: Collocation points
    """
    V = vandermonde_matrix(r)
    return np.linalg.inv(V @ V.T)


def diag_logical_mass_matrix(w):
    return np.diag(w)
