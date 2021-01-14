import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .spectral import inertial_coords
from .interpolate import lagrange_interpolate


def plot_dg(field, domain, show_element_boundaries=True, show_collocation_points=True, field_slice=None, slice_dim=None, slice_index=None, ax=None, label=None, color='darkorange', **plot_kwargs):
    """
    Plot a field on the computation domain.

    Parameters
    ----------
    field : str
        The name of the field to plot. Must be a scalar field.
    domain : domain.Domain
        The domain that carries the field data.
    show_element_boundaries : bool
        Show or hide element boundaries
    show_collocation_points : bool
        Show or hide collocation points
    field_slice : slice
        A slice that is applied to the field. Can be used to select which components of a vector or tensor field to plot.
    ax : The matplotlib axis to plot to
    **kwargs : Parameters are forwarded to matplotlib

    Parameters for 3D slice plots
    -----------------------------
    slice_dim : int
        The dimension to slice through
    slice_index : int
        The collocation point index to slice through in the slice_dim
    """

    if domain.dim == 1:
        if ax is None:
            ax = plt.gca()

        for e in domain.elements:
            field_data = getattr(e, field)
            if field_slice is not None:
                field_data = field_data[field_slice]

            if show_element_boundaries:
                for element_boundary in np.squeeze(e.extents):
                    boundary_handle = ax.axvline(
                        element_boundary, color='black')
            ax.plot(
                np.squeeze(e.inertial_coords),
                field_data,
                marker='.',
                ls='none',
                color=color,
                **plot_kwargs
            )
            if show_collocation_points:
                for collocation_point in np.squeeze(e.inertial_coords):
                    point_handle = ax.axvline(
                        collocation_point, color='black', ls='dotted', alpha=0.2)
            logical_space = np.linspace(-1, 1,
                                        int(200 / np.squeeze(domain.num_elements)))
            inertial_space = np.squeeze(
                inertial_coords([logical_space], e.extents))
            dg_handle = ax.plot(
                inertial_space,
                lagrange_interpolate(np.squeeze(
                    e.logical_coords), field_data)(logical_space),
                color=color,
                **plot_kwargs
            )[0]
        # Compose legend items
        handles, labels = [], []
        if label is not None:
            handles.append(dg_handle)
            labels.append(label)
        if show_element_boundaries:
            handles.append(boundary_handle)
            labels.append("Element boundaries")
        if show_collocation_points:
            handles.append(point_handle)
            labels.append("Collocation points")

        return handles, labels

    elif domain.dim == 2:
        # TODO
        if ax is None:
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection='3d')
        for e in domain.elements:
            ax.plot_surface(*e.inertial_coords, getattr(e, field))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(field)

    elif domain.dim == 3:
        # TODO
        assert slice_dim is not None, "You have to specify a slice_dim in 3D."
        assert slice_index is not None, "You have to specify a slice_index in 3D."
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        in_e_id = int(slice_index / domain.num_points[slice_dim])
        i = slice_index % domain.num_points[slice_dim]
        for e_id, e in domain.indexed_elements.items():
            if e_id[slice_dim] != in_e_id:
                continue
            x_d = np.mean(e.inertial_coords[slice_dim].take(i, axis=slice_dim))
            x = np.delete(e.inertial_coords, slice_dim, axis=0).take(i, axis=1 + slice_dim)
            u = np.take(getattr(e, field), i, axis=slice_dim)
            ax.plot_surface(*x, u)
        x_labels = ["x", "y", "z"]
        plt.title("Slice through ${}={:.2f}$".format(x_labels[slice_dim], x_d))
        u_all = domain.get_data(field)
        if np.min(u_all) != np.max(u_all):
            ax.set_zlim(np.min(u_all), np.max(u_all))
        ax.set_xlabel(np.delete(x_labels, slice_dim)[0])
        ax.set_ylabel(np.delete(x_labels, slice_dim)[1])
        ax.set_zlabel(field)

    else:
        raise NotImplementedError
