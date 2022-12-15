import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import LineString
import pathlib
import glob
import numpy as np
import math

RECON_DIR = '/home/tim/Dropbox/NIWA_UC/January_2021/Station_3_Grid/'

SAVE_2D = False

R = 6378.137  # Radius of earth in KM

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

def main():
    shape_files = glob.glob(RECON_DIR + '*.gpkg')
    print(shape_files)
    for sf in shape_files:
        if 'labelled' in sf:
            continue
        fn = sf.split('/')[-1].split('.')[0]
        gdf = gpd.read_file(sf)
        new_labels = []
        new_geom = []
        for name, linestring in zip(gdf.NAME, gdf.geometry):
            line_pnts = np.array(linestring.coords)
            length = 0
            if len(line_pnts) > 2:
                print("line with >2 pnts!")
            for i in range(len(line_pnts) - 1):
                len_2D = measure_chordlen(line_pnts[i, 1], line_pnts[i, 0], line_pnts[i+1, 1], line_pnts[i+1, 0])
                length += len_2D # math.sqrt(len_2D**2 + (line_pnts[i+1, 2] - line_pnts[i, 2])**2)
            label = fn + '_' + str(round(length, 4))
            print(label)
            label = '_'.join(label.split(' '))
            new_labels.append(label)
            if len(line_pnts) > 1:
                new_line = [xy[:2] for xy in list(linestring.coords)]
                new_geom.append(LineString(new_line))
            else:
                new_geom.append(linestring)
        gdf.NAME = new_labels
        labelled_str = '_labelled'
        if SAVE_2D:
            gdf.geometry = new_geom
            labelled_str += '_2D'
        gdf.to_file(RECON_DIR + '_'.join(fn.split(' ')) + labelled_str + '.gpkg')

if __name__ == '__main__':
    main()