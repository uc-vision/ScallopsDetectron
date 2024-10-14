from math import cos, radians, degrees, sin, asin, sqrt, atan2, atan
import numpy as np
import pyproj

R = 6370  # Radius of earth in KM at NZ (-35 latitude)

# Ellipsoid parameters: semi major axis in metres, reciprocal flattening.
GRS80 = 6378137, 298.257222100882711
WGS84 = 6378137, 298.257223563
clarke1880 = 6378249.145, 6356514.966


transformer_gd2gc = pyproj.Transformer.from_crs(
    {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
    {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
    )
def geodetic_to_geocentric_2(latitude, longitude, height):
    return transformer_gd2gc.transform(latitude, longitude, height)


def geodetic_to_geocentric(latitude, longitude, height):
    """Return geocentric (Cartesian) Coordinates x, y, z corresponding to
    the geodetic coordinates given by latitude and longitude (in
    degrees) and height above ellipsoid. The ellipsoid must be
    specified by a pair (semi-major axis, reciprocal flattening).
    """
    ellipsoid = WGS84
    φ = radians(latitude)
    λ = radians(longitude)
    sin_φ = sin(φ)
    a, rf = ellipsoid           # semi-major axis, reciprocal flattening
    e2 = 1 - (1 - 1 / rf) ** 2  # eccentricity squared
    n = a / sqrt(1 - e2 * sin_φ ** 2) # prime vertical radius
    r = (n + height) * cos(φ)   # perpendicular distance from z axis
    x = r * cos(λ)
    y = r * sin(λ)
    z = (n * (1 - e2) + height) * sin_φ
    return x, y, z


transformer_gc2gd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
def geocentric_to_geodetic(x, y, z):
    return transformer_gc2gd.transform(x, y, z, direction="INVERSE")


def convert_gps2local(datum, coords):
    local = coords.copy() - datum
    local[:, :2] = np.radians(local[:, :2])
    local[:, 1] = np.sin(local[:, 1])
    local[:, 0] = np.cos(np.radians(datum[1])) * np.sin(local[:, 0])
    local[:, :2] *= R * 1000
    return local

def convert_local2gps(datum, coords):
    # lon, lat
    gps = coords.copy()
    gps[:, :2] /= R * 1000
    gps[:, 1] = np.degrees(np.arcsin(gps[:, 1]))
    gps[:, 0] = np.degrees(np.arcsin(gps[:, 0] / (np.cos(np.radians(gps[:, 1] + datum[1])))))
    gps += datum
    return gps

def measure_arclen(lonlat1, lonlat2):  # generally used geo measurement function
    dLat = radians(lonlat2[1] - lonlat1[1])
    dLon = radians(lonlat2[0] - lonlat1[0])
    # calculates circle half chord length (squared) where hypotenuse is 1
    a = sin(dLat/2) ** 2 + cos(radians(lonlat1[1])) * cos(radians(lonlat2[1])) * sin(dLon/2) ** 2
    # 2 * atan2( Opp / Adj ) to find angular separation
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    # Calculate arc lengthqq
    d = R * c
    return d * 1000  # meters

def measure_chordlen(lonlat1, lonlat2):
    dLat = radians(lonlat2[1] - lonlat1[1])
    dLon = radians(lonlat2[0] - lonlat1[0])
    a = sin(dLat / 2) ** 2 + cos(radians(lonlat1[1])) * cos(radians(lonlat2[1])) * sin(dLon / 2) ** 2
    c = 2 * R * asin(sqrt(a))
    return c * 1000
