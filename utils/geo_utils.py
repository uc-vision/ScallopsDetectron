import math
import numpy as np

R = 6370  # Radius of earth in KM at NZ (-35 latitude)

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
    dLat = math.radians(lonlat2[1] - lonlat1[1])
    dLon = math.radians(lonlat2[0] - lonlat1[0])
    # calculates circle half chord length (squared) where hypotenuse is 1
    a = math.sin(dLat/2) ** 2 + math.cos(math.radians(lonlat1[1])) * math.cos(math.radians(lonlat2[1])) * math.sin(dLon/2) ** 2
    # 2 * atan2( Opp / Adj ) to find angular separation
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    # Calculate arc lengthqq
    d = R * c
    return d * 1000  # meters

def measure_chordlen(lonlat1, lonlat2):
    dLat = math.radians(lonlat2[1] - lonlat1[1])
    dLon = math.radians(lonlat2[0] - lonlat1[0])
    a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lonlat1[1])) * math.cos(math.radians(lonlat2[1])) * math.sin(dLon / 2) ** 2
    c = 2 * R * math.asin(math.sqrt(a))
    return c * 1000
