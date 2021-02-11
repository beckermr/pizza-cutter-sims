import numpy as np
import galsim


def gen_affine_wcs(
        *, rng, position_angle_range, dither_range,
        scale, scale_frac_std, shear_std,
        world_origin, origin):
    """Generate a random AffineTransform WCS.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG to use to generate the random WCS.
    position_angle_range : 2-tuple of floats
        The range of position angles to select from for rotating the image
        WCS coordinares.
    dither_range : 2-tuple of floats
        The lowest and highest dither in world coordinates.
    scale : float
        The mean pixel scale of the image,
    scale_frac_std : float
        The fractional variance in the generated image pixel scale.
    shear_std : float
        The standard deviation of the Gaussian shear put into the WCS.
    world_origin : galsim.PositionD
        The location of the origin of the image coordinate system in the
        world coordinate system.
    origin : galsim.PositionD
        The location of the origin of the world coordinate system in the
        image coordinate system. Note that the imaage origin is dithered if
        requested to keep the world origin fixed.

    Returns
    -------
    wcs : galsim.AffineTransform
        The randomly generated AffineTransform WCS object.
    """
    # an se wcs is generated from
    # 1) a pixel scale
    # 2) a shear
    # 3) a rotation angle
    # 4) a dither in the u,v plane of the location of the
    #    the image origin
    g1 = rng.normal() * shear_std
    g2 = rng.normal() * shear_std
    scale = (1.0 + rng.normal() * scale_frac_std) * scale
    theta = rng.uniform(
        low=position_angle_range[0],
        high=position_angle_range[1]) / 180.0 * np.pi
    dither_u = rng.uniform(
        low=dither_range[0],
        high=dither_range[1]) * scale
    dither_v = rng.uniform(
        low=dither_range[0],
        high=dither_range[1]) * scale
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    jac_matrix = scale * np.dot(
        galsim.Shear(g1=g1, g2=g2).getMatrix(),
        np.array([[costheta, -sintheta], [sintheta, costheta]])
        )
    dudx = jac_matrix[0, 0]
    dudy = jac_matrix[0, 1]
    dvdx = jac_matrix[1, 0]
    dvdy = jac_matrix[1, 1]
    dxdy = np.dot(np.linalg.inv(jac_matrix), np.array([dither_u, dither_v]))

    return galsim.AffineTransform(
        dudx, dudy, dvdx, dvdy,
        origin=origin + galsim.PositionD(x=dxdy[0], y=dxdy[1]),
        world_origin=world_origin,
    )
