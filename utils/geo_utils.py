import math
import numpy as np
R = 6378.137  # Radius of earth in KM

def convert_gps2local(datum, coords):
    local = coords.copy() - datum
    local[:, :2] = np.radians(local[:, :2])
    local[:, 1] = np.sin(local[:, 1])
    local[:, 0] = np.cos(np.radians(datum[1])) * np.sin(local[:, 0])
    local[:, :2] *= R * 1000
    return local

def convert_local2gps(datum, coords):
    gps = coords.copy()
    gps[:, :2] /= R * 1000
    gps[:, 1] = np.degrees(np.arcsin(gps[:, 1]))
    gps[:, 0] = np.degrees(np.arcsin(gps[:, 0] / (np.cos(np.radians(datum[1])))))
    gps += datum
    return gps

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