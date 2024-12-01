from typing import Any, Tuple

import mnn.vision.image_size


def _calculate_lower_bound(
    xc_norm: float, yc_norm: float, grid_Sx_norm: float, grid_Sy_norm: float
) -> Tuple[int, int]:
    lower_bound_x = int((xc_norm * 100) // (grid_Sx_norm * 100))
    lower_bound_y = int((yc_norm * 100) // (grid_Sy_norm * 100))
    return lower_bound_x, lower_bound_y


def calculate_grid(
    xc_norm: float,
    yc_norm: float,
    field_shape: mnn.vision.image_size.ImageSize,
    grid_S: mnn.vision.image_size.ImageSize,
) -> Tuple[int, int]:
    grid_Sx_norm = grid_S.width / field_shape.width
    grid_Sy_norm = grid_S.height / field_shape.height
    lower_bound_x, lower_bound_y = _calculate_lower_bound(
        xc_norm, yc_norm, grid_Sx_norm, grid_Sy_norm
    )
    return lower_bound_x, lower_bound_y


def twoD_grid_position_to_oneD_position(
    x: int, y: int, grid_S: mnn.vision.image_size.ImageSize
) -> int:
    return int(y * grid_S.width + x)


def oneD_position_to_twoD_grid_position(
    flattened_position: int, grid_S: mnn.vision.image_size.ImageSize
) -> Tuple[int, int]:
    y = flattened_position // grid_S.width
    x = flattened_position % grid_S.width
    return x, y


def calculate_coordinate_in_grid(
    xc_norm: float,
    yc_norm: float,
    field_shape: mnn.vision.image_size.ImageSize,
    grid_S: mnn.vision.image_size.ImageSize,
) -> Tuple[float, float]:
    grid_Sx_norm = grid_S.width / field_shape.width
    grid_Sy_norm = grid_S.height / field_shape.height
    n_x_grids, n_y_grids = _calculate_lower_bound(
        xc_norm, yc_norm, grid_Sx_norm, grid_Sy_norm
    )
    lower_bound_x = n_x_grids * grid_Sx_norm
    lower_bound_y = n_y_grids * grid_Sy_norm

    in_grid_x = (xc_norm - lower_bound_x) / grid_Sx_norm
    in_grid_y = (yc_norm - lower_bound_y) / grid_Sy_norm
    return in_grid_x, in_grid_y


def calculate_real_coordinate(
    xc_norm_in_grid: float,
    yc_norm_in_grid: float,
    grid_position_x: int,
    grid_position_y: int,
    grid_shape: mnn.vision.image_size.ImageSize,
    field_shape: mnn.vision.image_size.ImageSize,
) -> Tuple[float, float]:
    xc_in_grid = xc_norm_in_grid * grid_shape.width
    yc_in_grid = yc_norm_in_grid * grid_shape.height

    xc = xc_in_grid + grid_position_x * grid_shape.width
    yc = yc_in_grid + grid_position_y * grid_shape.height
    return xc / field_shape.width, yc / field_shape.height
