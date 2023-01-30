import math
import numpy as np
R = 6378.137  # Radius of earth in KM

def convert_gps2local(datum, coords):
    local = coords - datum
    local[:, :2] = np.radians(local[:, :2])
    local[:, 1] = 2 * np.sin(local[:, 1] / 2)
    local[:, 0] = 2 * np.cos(local[:, 1] / 2) * np.sin(local[:, 0] / 2)
    local[:, :2] *= R * 1000
    return local

def convert_local2gps(datum, coords):
    gps = coords
    gps[:, :2] = coords[:, :2] / (R * 1000)
    gps[:, 1] = np.degrees(2 * np.arcsin(gps[:, 1] / 2))
    gps[:, 0] = np.degrees(2 * np.arcsin(gps[:, 0] / (2 * np.cos(gps[:, 1] / 2))))
    gps = coords + datum
    return gps

def convert_gpsvec_m(lat1, lon1, lat2, lon2):
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    dy = 2 * np.sin(dLat / 2)
    dx = 2 * np.cos(np.radians((lat1 + lat2) / 2)) * np.sin(dLon / 2)
    return R * np.array([dx, dy]) * 1000

def measure_arclen(lat1, lon1, lat2, lon2):  # generally used geo measurement function
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    # calculates circle half chord length (squared) where hypotenuse is 1
    a = math.sin(dLat/2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2) ** 2
    # 2 * atan2( Opp / Adj ) to find angular separation
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    # Calculate arc length
    d = R * c
    return d * 1000  # meters

def measure_chordlen(lat1, lon1, lat2, lon2):
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) ** 2
    c = 2 * R * math.sqrt(a)
    return c * 1000