def mm3_ml(volume_mm3):
    """Converts mm³ volume to ml

    Args:
        volume_mm3 (float): volume in mm³

    Returns:
        float: volume in ml
    """
    return volume_mm3 / 1000


def mm_m(length_mm):
    """Converts mm lengh to m

    Args:
        length_mm (float): length in mm

    Returns:
        float: length in m
    """
    return length_mm / 1000


def m_mm(length_m):
    """Converts m lengh to mm

    Args:
        length_m (float): length in m

    Returns:
        float: length in mm
    """
    return length_m * 1000


def mm2_m2(area_mm2):
    """Converts mm² area to m²

    Args:
        area_mm2 (float): area in mm²

    Returns:
        float: area in m²
    """
    return area_mm2 / 1000000


def mm3_m3(volume_mm3):
    """Converts mm³ volume to m³

    Args:
        volume_mm3 (float): volume in mm³

    Returns:
        float: volume in m³
    """
    return volume_mm3 / 1000000000


def m3_ml(volume_m3):
    """Converts m³ volume to ml

    Args:
        volume_mm3 (float): volume in m³

    Returns:
        float: volume in ml
    """
    return volume_m3 * 1000000


def ml_m3(volume_ml):
    """Converts ml volume to m³

    Args:
        volume_mm3 (float): volume in ml

    Returns:
        float: volume in m³
    """
    return volume_ml / 1000000
