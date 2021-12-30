import cv2
import scipy
import skimage
import time

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import SimpleITK as sitk

from scipy import sparse
from skimage.morphology._util import _raveled_offsets_and_distances
from skimage.util._map_array import map_array
from skimage.graph._graph import _weighted_abs_diff
from typing import Optional


def pixel_graph(
    image, *, mask=None, edge_function=None, max_g=None, weights_r=None,
    weights_u=None, ilm_image=None, connectivity=1, spacing=None, purpose=None
):
    """
    REFACTORED VERSION OF skimage.graph._graph.pixel_graph

    Create an adjacency graph of pixels in an image.
    Pixels where the mask is True are nodes in the returned graph, and they are
    connected by edges to their neighbors according to the connectivity
    parameter. By default, the *value* of an edge when a mask is given, or when
    the image is itself the mask, is the euclidean distance betwene the pixels.
    However, if an int- or float-valued image is given with no mask, the value
    of the edges is the absolute difference in intensity between adjacent
    pixels, weighted by the euclidean distance.
    Parameters
    ----------
    image : array
        The input image. If the image is of type bool, it will be used as the
        mask as well.
    mask : array of bool
        Which pixels to use. If None, the graph for the whole image is used.
    edge_function : callable
        A function taking an array of pixel values, and an array of neighbor
        pixel values, and an array of distances, and returning a value for the
        edge. If no function is given, the value of an edge is just the
        distance.
    connectivity : int
        The square connectivity of the pixel neighborhood: the number of
        orthogonal steps allowed to consider a pixel a neigbor. See
        `scipy.ndimage.generate_binary_structure` for details.
    spacing : tuple of float
        The spacing between pixels along each axis.
    Returns
    -------
    graph : scipy.sparse.csr_matrix
        A sparse adjacency matrix in which entry (i, j) is 1 if nodes i and j
        are neighbors, 0 otherwise.
    nodes : array of int
        The nodes of the graph. These correspond to the raveled indices of the
        nonzero pixels in the mask.
    """
    if image.dtype == bool and mask is None:
        mask = image
    if mask is None and edge_function is None:
        mask = np.ones_like(image, dtype=bool)
        edge_function = _weighted_abs_diff
    # Main modification from the original version
    if mask is None and edge_function is not None:
        mask = np.ones_like(image, dtype=bool)

    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    nodes_padded = np.flatnonzero(padded)
    neighbor_offsets_padded, distances_padded = _raveled_offsets_and_distances(
        padded.shape, connectivity=connectivity, spacing=spacing
    )
    neighbors_padded = nodes_padded[:, np.newaxis] + neighbor_offsets_padded
    neighbor_distances_full = np.broadcast_to(
        distances_padded, neighbors_padded.shape
    )
    nodes = np.flatnonzero(mask)
    nodes_sequential = np.arange(nodes.size)

    neighbors = map_array(neighbors_padded, nodes_padded, nodes)
    neighbors_mask = padded.reshape(-1)[neighbors_padded]
    num_neighbors = np.sum(neighbors_mask, axis=1)
    indices = np.repeat(nodes, num_neighbors)
    indices_sequential = np.repeat(nodes_sequential, num_neighbors)
    neighbor_indices = neighbors[neighbors_mask]
    neighbor_distances = neighbor_distances_full[neighbors_mask]
    neighbor_indices_sequential = map_array(
        neighbor_indices, nodes, nodes_sequential
    )

    if edge_function is None:
        data = neighbor_distances
    # Main modification 2 from the original version
    elif purpose == 'line_segmentation':
        image_r = image.reshape(-1)
        shape = image.shape
        weights_r_flat = weights_r.reshape(-1)
        try:
            weights_u_flat = weights_u.reshape(-1)
        except Exception as e:
            weights_u_flat = weights_r_flat
        data = edge_function(
            image_r[indices], image_r[neighbor_indices], max_g, weights_r_flat[indices],
            weights_u_flat[indices], indices, ilm_image, shape
        )
    else:
        image_r = image.reshape(-1)
        image_r[indices]
        data = edge_function(image_r[indices], image_r[neighbor_indices], neighbor_distances)

    m = nodes_sequential.size
    mat = sparse.coo_matrix(
            (data, (indices_sequential, neighbor_indices_sequential)),
            shape=(m, m)
            )
    graph = mat.tocsr()

    return graph, nodes, indices


def gker(shape: tuple = (3, 3), sigma: float = 0.5):
    """
    Generates a 2D gaussian mask.
    Args:
        shape (tuple, optional): Shape of the kernel. Defaults to (3,3).
        sigma (float, optional): Std of the gaussian. Defaults to 0.5.

    Returns:
        np.ndarray: gaussian kernel
    """
    # Get the center and generate a grid of gaussian values
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m: (m + 1), -n:(n + 1)]
    h = np.exp(-(x**2 + y**2) / (2*(sigma**2)))
    # Truncate small values
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    # Renormalize after truncation
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def region_extractor(
    manufacturer: str, image: np.ndarray, reference: Optional[np.ndarray] = None
):
    """
    Eliminates the cero valued rows of the image and if needed, extracts
    the coarse region of the retina from an oct slice.
    Performs thresholding using the histogram of the image to define
    the best threshold and then morphological operations to eliminate
    background noise and extend the "retina" region.

    Args:
        manufacturer (str): name of the manufacturer.
            Valid ones: Topcon, Cirrus, Spectralis
        image (np.ndarray): Oct slice to be processed.
        reference (Optional[np.ndarray]): Reference image associated
            with the oct slice to be processed.
    Return:
        image (np.ndarray): Cropped version of oct slice.
        reference: If provided, cropped version of the reference image.
    """
    # Determine the threshold and the number of dilations according
    # to the manufacturer, this is obtained by experimentation
    manufacturer = manufacturer.lower()
    if manufacturer == 'topcon':
        threshold_offset = 10
        dilations = 10
    elif manufacturer == 'cirrus':
        threshold_offset = 15
        dilations = 20
    elif manufacturer == 'spectralis':
        threshold_offset = 25
        dilations = 8
#         cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
#         image = image.astype('uint8')
    else:
        raise ValueError(
            f'Manufacturer {manufacturer} not in the allowed ones, try again'
        )

    # Remove cero valued rows due to artifacts
    indx = np.where(np.sum(image[:,:], axis=1) > 0)
    image = image[np.sum(image[:,:],axis=1) > 0, :]
    if reference is not None:
        reference = reference[np.min(indx):np.max(indx), :]

    # Remove unnecesary background rows
    # Binarize
    [frec, val] = np.histogram(image, bins=30)
    thr = val[np.argmax(frec)] + threshold_offset
    mask = np.where(image > thr, 1, 0)
    # Morphological ops
    mask = skimage.morphology.erosion(mask, footprint=np.ones((5,5)))
    mask = skimage.morphology.opening(mask)
    for i in range(dilations):
        mask = skimage.morphology.dilation(mask, footprint=np.ones((11,11)))
    # Region extraction
    indx = np.where(np.sum(mask[:, :], axis=1)>0)
    image = image[np.min(indx):np.max(indx), :]
    if reference is not None:
        reference = reference[np.min(indx):np.max(indx), :]
        return image, reference
    return image

