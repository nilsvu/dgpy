import numpy as np
import itertools
import logging
import functools

from .spectral import lgl_points, lgl_weights, inertial_coords, logical_coords
from .plot import *

class Face:
    def __init__(self, element, dimension, direction, is_exterior=False):
        self.element = element
        self.dimension = dimension
        self.direction = direction
        self.is_exterior = is_exterior

        self.dim = element.dim - 1
        self.extents = np.delete(element.extents, dimension, axis=0)
        self.num_points = np.delete(element.num_points, dimension)
        self.collocation_points = np.delete(
            element.collocation_points, dimension, axis=0)
        self.quadrature_weights = np.delete(
            element.quadrature_weights, dimension, axis=0)

        self.logical_coords = np.array(np.meshgrid(
            *self.collocation_points, indexing='ij'))
        self.inertial_coords = inertial_coords(
            self.logical_coords, self.extents)

        self.inertial_to_logical_jacobian = np.delete(np.delete(
            element.inertial_to_logical_jacobian, dimension, axis=0), dimension, axis=1)
        self.inertial_to_logical_jacobian_det = np.linalg.det(
            self.inertial_to_logical_jacobian)

    def __repr__(self):
        if self.dim == 0 and self.dimension == 0 or self.dim > 0 and self.dimension == 1:
            if self.direction == -1:
                return "<|" if not self.is_exterior else "<"
            elif self.direction == 1:
                return "|>" if not self.is_exterior else ">"
        elif self.dim == 0 and self.dimension == 1 or self.dim > 0 and self.dimension == 0:
            if self.direction == -1:
                return "|^|" if not self.is_exterior else "^"
            elif self.direction == 1:
                return "|v|" if not self.is_exterior else "v"
        elif self.dimension == 2:
            if self.direction == -1:
                return "|x|" if not self.is_exterior else "x"
            elif self.direction == 1:
                return "|o|" if not self.is_exterior else "o"
        return "?"

    def is_in(self, category):
        return \
            category == 'any' \
            or category == 'internal' and not (self.is_exterior or self.opposite_face.is_exterior) \
            or category == 'external' and self.opposite_face.is_exterior \
            or category == 'exterior' and self.is_exterior \
            or category == 'interior' and not self.is_exterior

    def slice_index(self):
        if not self.is_exterior:
            return 0 if self.direction == -1 else -1
        else:
            return 0 if self.direction == 1 else -1

    def get_normal(self):
        normal = np.zeros((self.dim + 1, *self.num_points))
        normal[self.dimension, ...] = self.direction
        return normal

    def normal_dot(self, v):
        return np.einsum('d...,d...', self.get_normal(), v)

    def normal_times(self, u):
        face_indices = ''.join(['a', 'b', 'c'][:self.dim])
        return np.einsum('i' + face_indices + ',...' + face_indices + '->i...' + face_indices, self.get_normal(), u)

    def jump(self, field):
        """
        The exterior minus the interior field value.

        Reduces to the interior field value if no exterior face data is available.

        Note: Does not include the face normal.
        """
        field_interior = getattr(self, field)
        field_exterior = getattr(self.opposite_face, field)
        if field_exterior is not None:
            return field_interior - field_exterior
        else:
            return field_interior

    def average(self, field):
        """
        The average of the interior and exterior field values.

        Reduces to the interior field value if no exterior face data is available.
        """
        field_interior = getattr(self, field)
        field_exterior = getattr(self.opposite_face, field)
        if field_exterior is not None:
            return (field_interior + field_exterior) / 2
        else:
            return field_interior


class Element:
    logger = logging.getLogger('Element')

    def __init__(self, extents, num_points):
        logger = logging.getLogger('Domain.Creation')

        extents = np.asarray(extents)
        self.dim = extents.shape[0]

        logger.debug("Creating element with extents={}, num_points={}".format(
            extents, num_points))
        self.extents = extents
        self.num_points = num_points
        self.collocation_points = [lgl_points(N) for N in num_points]
        self.quadrature_weights = [lgl_weights(N) for N in num_points]

        self.logical_coords = np.array(np.meshgrid(
            *self.collocation_points, indexing='ij'))
        self.inertial_coords = inertial_coords(
            self.logical_coords, self.extents)

        self.inertial_to_logical_jacobian = np.diag(
            np.squeeze(np.diff(extents, axis=-1), axis=-1) / 2)
        self.inertial_to_logical_jacobian_det = np.linalg.det(
            self.inertial_to_logical_jacobian)

        self.indexed_faces = {}
        for d in range(self.dim):
            for direction in (-1, 1):
                face = Face(self, dimension=d, direction=direction)
                self.indexed_faces[(d, direction, False)] = face
        self.faces = self.indexed_faces.values()

    def __repr__(self):
        return repr(self.id)

    def get_internal_faces(self):
        """Faces to other elements"""
        return (face for face in self.faces if face.is_in('internal'))

    def get_external_faces(self):
        """Faces to the domain boundary"""
        return (face for face in self.faces if face.is_in('external'))

    def get_exterior_faces(self):
        """\"Ghost\"-faces of the non-existent elements across the domain boundary (used to impose boundary conditions)"""
        return (face for face in self.faces if face.is_in('exterior'))

    def get_interior_faces(self):
        """All non-exterior faces, i.e. the union of internal and external faces."""
        return (face for face in self.faces if face.is_in('interior'))

    def slice_to_faces(self, field, category='any'):
        field_in_volume = getattr(self, field)
        for face in self.faces:
            if face.is_in(category):
                field_on_slice = np.take(field_in_volume, face.slice_index(
                ), axis=face.dimension + (field_in_volume.ndim - self.dim))
                if face.is_exterior:
                    field_on_slice *= -1  # Dirichlet conditions
            else:
                field_on_slice = None
            setattr(face, field, field_on_slice)

    def get_field_in(self, field, sub_extents):
        xis, field_values = [], []
        for point in itertools.product(*[range(self.num_points[d]) for d in range(self.dim)]):
            x = self.inertial_coords
            u = getattr(self, field)
            valence = u.ndim - self.dim
            for d in range(self.dim):
                x = np.take(x, point[d], axis=1)
                u = np.take(u, point[d], axis=valence)
            choose_point = True
            for d in range(self.dim):
                if x[d] < sub_extents[d][0] or x[d] > sub_extents[d][1]:
                    choose_point = False
            if choose_point:
                xis.append(logical_coords(x, sub_extents))
                field_values.append(u)
        return xis, field_values


class Domain:
    """A D-dimensional computational domain"""

    logger = logging.getLogger('Domain')

    def __init__(self, extents=None, num_elements=None, num_points=None, elements=None, element_ids=None):
        """
        Create a D-dimensional computational domain.

        Parameters
        ----------
        extents : (D, 2) array_like
            The lower and upper bound of the domain in each dimension.
        num_elements : int or (D,) array_like
            The number of elements in each dimension.
        num_points : int or (D,) array_like
            The number of points in an element in each dimension.

        Alternative Parameters
        ----------------------
        elements : A list of elements that compose the domain. Must also provide `element_ids`. All previous parameters are ignored.
        element_ids : The ID for each element in `elements`. The ID is a tuple of indices that define the element's position on a rectangular grid.
        """
        logger = logging.getLogger('Domain.Creation')

        if elements is None:
            logger.debug("Creating domain with extents={}, num_elements={}, num_points={}".format(
                extents, num_elements, num_points))
            extents = np.asarray(extents)
            if extents.ndim == 1:
                extents = np.expand_dims(extents, axis=0)
            self.dim = extents.shape[0]
            self.extents = extents
            logger.debug("Determined dimension is {}".format(self.dim))

            num_elements = np.asarray(num_elements)
            try:
                assert len(
                    num_elements) == self.dim, "num_elements does not match the dimension"
            except TypeError:
                num_elements = np.repeat(num_elements, self.dim)
            self.num_elements = num_elements

            num_points = np.asarray(num_points)
            try:
                assert len(
                    num_points) == self.dim, "num_points does not match the dimension"
            except TypeError:
                num_points = np.repeat(num_points, self.dim)
            self.num_points = tuple(num_points)

            domain_size = np.squeeze(np.diff(extents))
            element_size = domain_size / num_elements

            # Create elements
            self.indexed_elements = {}
            for element_id in itertools.product(*[range(num_elements[d]) for d in range(self.dim)]):
                element_extents = [
                    extents[d][0] + element_size[d] *
                    np.array([element_id[d], element_id[d] + 1])
                    for d in range(self.dim)
                ]
                e = Element(extents=element_extents,
                            num_points=self.num_points)
                e.id = element_id
                self.indexed_elements[e.id] = e
            self.elements = self.indexed_elements.values()
        else:
            logger.debug(
                "Creating domain with {} elements".format(len(elements)))

            all_extents = np.array([e.extents for e in elements])
            self.extents = np.transpose(np.array(
                [np.min(all_extents[:, :, 0], axis=0), np.max(all_extents[:, :, 1], axis=0)]))
            self.dim = self.extents.shape[0]
            logger.debug("Determined dimension is {}".format(self.dim))

            self.indexed_elements = {}
            for element_id, e in zip(element_ids, elements):
                e.id = element_id
                self.indexed_elements[e.id] = e
            self.elements = self.indexed_elements.values()

            self.num_elements = np.max(np.asarray(element_ids), axis=0)

        # Connect faces
        for e in self.elements:
            for face in list(e.faces):
                d, direction = face.dimension, face.direction
                if not face.is_exterior:
                    neighbour_id = list(e.id)
                    neighbour_id[d] += direction
                    if neighbour_id[d] in range(0, self.num_elements[d]):
                        neighbour = self.indexed_elements[tuple(neighbour_id)]
                        face.opposite_face = neighbour.indexed_faces[(
                            d, -direction, False)]
                    else:
                        ghost_face = Face(
                            e, dimension=d, direction=-direction, is_exterior=True)
                        e.indexed_faces[(d, -direction, True)] = ghost_face
                        face.opposite_face = ghost_face
                        ghost_face.opposite_face = face

    def get_total_num_points(self):
        return self.reduce(
            lambda a, b: a + b,
            lambda e: np.product(e.num_points),
            0
        )

    def get_inertial_coords(self):
        """The inertial coordinates across all elements"""
        inertial_coords = self.dim * [[]]
        for e in self.elements:
            for d in range(self.dim):
                inertial_coords[d] = inertial_coords[d] + \
                    list(e.inertial_coords[d].flatten())
        return np.array(inertial_coords)

    def get_restriction_operator(self, e_center, overlap):
        num_points = e_center.num_points[0]
        subdomain_num_points = num_points + (overlap if e_center.id[0] != 0 else 0) + (
            overlap if e_center.id[0] != self.num_elements[0] - 1 else 0)
        R = np.zeros((subdomain_num_points, self.get_total_num_points()))
        i_start = e_center.id[0] * num_points - \
            (overlap if e_center.id[0] != 0 else 0)
        i_end = (e_center.id[0] + 1) * num_points + \
            (overlap if e_center.id[0] != self.num_elements[0] - 1 else 0)
        R[:, i_start:i_end] = np.eye(subdomain_num_points)
        return R

    def get_extended_logical_coords(self, e_center, overlap):
        extended_xi = []
        if e_center.id[0] != 0:
            extended_xi += list((e_center.logical_coords[0] - 2)[
                                e_center.num_points[0]-overlap:])
        extended_xi += list(e_center.logical_coords[0])
        if e_center.id[0] != self.num_elements[0] - 1:
            extended_xi += list((e_center.logical_coords[0] + 2)[:overlap])
        return [np.asarray(extended_xi)]

    def get_field_in(self, field, sub_extents):
        logical_coords, field_values = [], []
        for e in self.elements:
            xis, us = e.get_field_in(field, sub_extents)
            logical_coords += xis
            field_values += us
        return logical_coords, field_values

    def elements_in(self, extents):
        extents = np.asarray(extents)
        return (e for e in self.elements if np.all(np.logical_and(np.asarray(e.extents)[:,0] >= extents[:,0], np.asarray(e.extents)[:,1] <= extents[:,1])))

    def set_data(self, data, fields, fields_valence=None, order='C', **kwargs):
        """
        Distributes the `data` to all elements.

        Parameters
        ----------
        data : function or array_like
            If `data` is a function, it is expected to be callable with the (D, ...) inertial coordinates and the `kwargs`. It represents a single field and its valence is inferred from the function output.
            If `data` is an array, it is expected to be flattened over all elements and fields. Each chunk of the data corresponds to a field within an element. The chunk data is reshaped to `field_valence * (D,) + element.num_points`. The data is expected to be in the same order as returned by `self.elements`. Within the data for an element, the chunks are ordered by the field they correspond to in the list `fields`.
        fields : str or list
            The name of a field or a list of multiple fields. The data is set on the elements as attributes with the field names.
        fields_valence : int or list
            The of the field, or a list of valences for each field in `fields`. Defaults to 0 (scalar fields).
        order: str
            Layout of flattened array `data`. See documentation for numpy.reshape or numpy.flatten for possible values.
        """
        if isinstance(fields, str):
            fields = [fields]
        if fields_valence is not None and np.asarray(fields_valence).ndim == 0:
            fields_valence = [fields_valence]
        assert fields_valence is None or len(fields_valence) == len(
            fields), "Specify a valence for each field"

        try:
            _ = len(data)
            is_flattened = True
            data = np.asarray(data)
            if fields_valence is None:
                fields_valence = (0,) * len(fields)
        except TypeError:
            assert len(
                fields) == 1, "A data function must represent a single field."
            assert fields_valence is None, "The field valence is inferred from the function output."
            data = data(self.get_inertial_coords(), **kwargs)
            fields_valence = (data.ndim - 1,)
            is_flattened = False

        k = 0
        for e in self.elements:
            for i, field in enumerate(fields):
                valence_shape = fields_valence[i] * (self.dim,)
                num_dofs = self.dim**fields_valence[i]
                num_points = np.prod(e.num_points)
                if is_flattened:
                    element_data = np.reshape(np.reshape(data[k:k + num_points * num_dofs],
                        valence_shape + (np.product(e.num_points),)),
                            valence_shape + tuple(e.num_points), order=order)
                    k += num_points * num_dofs
                else:
                    element_data = np.reshape(
                        data[..., k:k + num_points], (*valence_shape, *e.num_points))
                    k += num_points
                setattr(e, field, element_data)

    def get_data(self, fields, order='C'):
        """Retrieve the datasets of the specified fields as a flat array."""
        if isinstance(fields, str):
            fields = [fields]
        return np.concatenate([
            np.concatenate([
                getattr(e, field).flatten(order=order)
                for field in fields])
            for e in self.elements])

    def apply(self, operator, field, result_field, *args, **kwargs):
        for e in self.elements:
            setattr(e, result_field, operator(
                getattr(e, field), e, *args, **kwargs))

    def reduce(self, reduction_operator, element_operator, initial):
        return functools.reduce(
            reduction_operator,
            map(element_operator, self.elements),
            initial
        )

    def plot(self, field, *args, **kwargs):
        """Alias for plot.plot_dg"""
        return plot_dg(field, self, *args, **kwargs)
