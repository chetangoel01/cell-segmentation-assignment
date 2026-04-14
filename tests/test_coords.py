import numpy as np
from src.coords import pixel_to_um, parse_boundary_polygon, spots_in_polygon


def test_pixel_to_um_origin():
    x_um, y_um = pixel_to_um(0, 0, fov_x=100.0, fov_y=200.0, pixel_size=0.109)
    assert abs(x_um - 100.0) < 1e-6
    assert abs(y_um - 200.0) < 1e-6


def test_pixel_to_um_offset():
    x_um, y_um = pixel_to_um(10, 20, fov_x=0.0, fov_y=0.0, pixel_size=0.109)
    assert abs(x_um - 1.09) < 1e-6
    assert abs(y_um - 2.18) < 1e-6


def test_parse_boundary_polygon_valid():
    xs = "1.0,2.0,2.0,1.0"
    ys = "1.0,1.0,2.0,2.0"
    poly = parse_boundary_polygon(xs, ys)
    assert poly is not None
    assert poly.is_valid


def test_parse_boundary_polygon_empty():
    poly = parse_boundary_polygon("", "")
    assert poly is None


def test_spots_in_polygon():
    xs = "0.0,4.0,4.0,0.0"
    ys = "0.0,0.0,4.0,4.0"
    poly = parse_boundary_polygon(xs, ys)
    spot_x = np.array([2.0, 10.0])  # inside, outside
    spot_y = np.array([2.0, 10.0])
    inside = spots_in_polygon(spot_x, spot_y, poly)
    assert inside.tolist() == [True, False]

