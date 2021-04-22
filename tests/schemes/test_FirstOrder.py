import unittest

import numpy as np
import numpy.testing as npt
from dgpy.domain import Domain, BoundaryCondition
from dgpy.boundary_conditions.Zero import Zero
from dgpy.schemes.FirstOrder import DgOperator
from dgpy.systems import Poisson


class TestFirstOrderScheme(unittest.TestCase):
    def test_regression_1d(self):
        domain = Domain(extents=[(-0.5, 1.5)], num_elements=4, num_points=4)
        boundary_conditions = [(
            Zero(BoundaryCondition.DIRICHLET),
            Zero(BoundaryCondition.NEUMANN),
        )]
        A = DgOperator(domain=domain,
                       system=Poisson,
                       boundary_conditions=boundary_conditions,
                       formulation='flux',
                       scheme='strong',
                       numerical_flux='ip',
                       massive=False,
                       mass_lumping=True,
                       penalty_parameter=1.5,
                       lifting_scheme='mass_matrix',
                       storage_order='F')
        domain.indexed_elements[(0, )].u = np.array([
            0.6964691855978616, 0.28613933495037946, 0.2268514535642031,
            0.5513147690828912
        ])
        domain.indexed_elements[(1, )].u = np.array([
            0.7194689697855631, 0.42310646012446096, 0.9807641983846155,
            0.6848297385848633
        ])
        domain.indexed_elements[(2, )].u = np.array([
            0.7194689697855631, 0.42310646012446096, 0.9807641983846155,
            0.6848297385848633
        ])
        domain.indexed_elements[(3, )].u = np.array([
            0.48093190148436094, 0.3921175181941505, 0.3431780161508694,
            0.7290497073840416
        ])
        u = domain.get_data('u', storage_order='F')
        Au = A @ u
        domain.set_data(Au, 'Au', storage_order='F')
        npt.assert_allclose(domain.indexed_elements[(0, )].Au, [
            1785.7616039198579, 41.55242721423977, -41.54933376905333,
            -80.83628748342855
        ])

    def test_regression_2d(self):
        domain = Domain(extents=[(-0.5, 1.5), (0., 1.)],
                        num_elements=[2, 2],
                        num_points=[3, 2])
        boundary_conditions = 2 * [(
            Zero(BoundaryCondition.DIRICHLET),
            Zero(BoundaryCondition.DIRICHLET),
        )]
        A = DgOperator(domain=domain,
                       system=Poisson,
                       boundary_conditions=boundary_conditions,
                       formulation='flux',
                       scheme='strong',
                       numerical_flux='ip',
                       massive=False,
                       mass_lumping=True,
                       penalty_parameter=1.5,
                       lifting_scheme='mass_matrix',
                       storage_order='F')
        domain.indexed_elements[(0, 1)].u = np.array([
            0.9807641983846155, 0.6848297385848633, 0.48093190148436094,
            0.3921175181941505, 0.3431780161508694, 0.7290497073840416
        ])
        domain.indexed_elements[(0, 0)].u = np.array([
            0.6964691855978616, 0.28613933495037946, 0.2268514535642031,
            0.5513147690828912, 0.7194689697855631, 0.42310646012446096
        ])
        domain.indexed_elements[(1, 1)].u = np.array([
            0.5315513738418384, 0.5318275870968661, 0.6344009585513211,
            0.8494317940777896, 0.7244553248606352, 0.6110235106775829
        ])
        domain.indexed_elements[(1, 0)].u = np.zeros((3, 2))
        u = domain.get_data('u', storage_order='F')
        Au = A @ u
        domain.set_data(Au, 'Au', storage_order='F')
        npt.assert_allclose(
            domain.indexed_elements[(0, 1)].Au,
            np.array([
                203.56354715945108, 9.40868981828554, -2.818657740285368,
                111.70107437132107, 35.80427083086546, 65.53029015630551
            ]).reshape((3, 2), order='F'))

    def test_regression_3d(self):
        domain = Domain(extents=[(-0.5, 1.5), (0., 1.), (-1., 3.)],
                        num_elements=[2, 2, 2],
                        num_points=[2, 3, 4])
        boundary_conditions = 3 * [(
            Zero(BoundaryCondition.DIRICHLET),
            Zero(BoundaryCondition.DIRICHLET),
        )]
        A = DgOperator(domain=domain,
                       system=Poisson,
                       boundary_conditions=boundary_conditions,
                       formulation='flux',
                       scheme='strong',
                       numerical_flux='ip',
                       massive=False,
                       mass_lumping=True,
                       penalty_parameter=1.5,
                       lifting_scheme='mass_matrix',
                       storage_order='F')
        domain.indexed_elements[(0, 0, 0)].u = np.array([
            0.6964691855978616, 0.28613933495037946, 0.2268514535642031,
            0.5513147690828912, 0.7194689697855631, 0.42310646012446096,
            0.9807641983846155, 0.6848297385848633, 0.48093190148436094,
            0.3921175181941505, 0.3431780161508694, 0.7290497073840416,
            0.4385722446796244, 0.05967789660956835, 0.3980442553304314,
            0.7379954057320357, 0.18249173045349998, 0.17545175614749253,
            0.5315513738418384, 0.5318275870968661, 0.6344009585513211,
            0.8494317940777896, 0.7244553248606352, 0.6110235106775829
        ])
        domain.indexed_elements[(1, 0, 0)].u = np.array([
            0.15112745234808023, 0.39887629272615654, 0.24085589772362448,
            0.34345601404832493, 0.5131281541990022, 0.6666245501640716,
            0.10590848505681383, 0.13089495066408074, 0.32198060646830806,
            0.6615643366662437, 0.8465062252707221, 0.5532573447967134,
            0.8544524875245048, 0.3848378112757611, 0.31678789711837974,
            0.3542646755916785, 0.17108182920509907, 0.8291126345018904,
            0.3386708459143266, 0.5523700752940731, 0.578551468108833,
            0.5215330593973323, 0.002688064574320692, 0.98834541928282
        ])
        domain.indexed_elements[(0, 1, 0)].u = np.array([
            0.5194851192598093, 0.6128945257629677, 0.12062866599032374,
            0.8263408005068332, 0.6030601284109274, 0.5450680064664649,
            0.3427638337743084, 0.3041207890271841, 0.4170222110247016,
            0.6813007657927966, 0.8754568417951749, 0.5104223374780111,
            0.6693137829622723, 0.5859365525622129, 0.6249035020955999,
            0.6746890509878248, 0.8423424376202573, 0.08319498833243877,
            0.7636828414433382, 0.243666374536874, 0.19422296057877086,
            0.5724569574914731, 0.09571251661238711, 0.8853268262751396
        ])
        domain.indexed_elements[(0, 0, 1)].u = np.array([
            0.7224433825702216, 0.3229589138531782, 0.3617886556223141,
            0.22826323087895561, 0.29371404638882936, 0.6309761238544878,
            0.09210493994507518, 0.43370117267952824, 0.4308627633296438,
            0.4936850976503062, 0.425830290295828, 0.3122612229724653,
            0.4263513069628082, 0.8933891631171348, 0.9441600182038796,
            0.5018366758843366, 0.6239529517921112, 0.11561839507929572,
            0.3172854818203209, 0.4148262119536318, 0.8663091578833659,
            0.2504553653965067, 0.48303426426270435, 0.985559785610705
        ])
        for element_id in [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
            domain.indexed_elements[element_id].u = np.zeros((2, 3, 4))
        u = domain.get_data('u', storage_order='F')
        Au = A @ u
        domain.set_data(Au, 'Au', storage_order='F')
        npt.assert_allclose(
            domain.indexed_elements[(0, 0, 0)].Au,
            np.array([
                618.6142450194411, 269.8213716601356, 49.33225265133292,
                103.71967654882658, 219.4353476547795, -14.237023651828594,
                731.9842766450536, 490.9303825979318, 32.18932195031287,
                13.87090223491767, -13.954381736466516, 130.61721549991918,
                331.75024822120696, 55.511704965231125, 17.52350289937635,
                23.878697549520762, -183.34493489042083, -171.66677910143915,
                390.19201603025016, 410.25585855100763, 53.690124372228034,
                82.23683297149915, 53.091014251828675, 117.36921898587735
            ]).reshape((2, 3, 4), order='F'))
