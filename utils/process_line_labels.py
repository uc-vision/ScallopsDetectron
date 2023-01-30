import geopandas as gpd
import rasterio
from shapely.geometry import LineString
import glob
import numpy as np
import math
from utils import geo_utils

HOME_DIR = '/local'  # /home/tim
RECON_DIR = HOME_DIR + '/Dropbox/NIWA_UC/January_2021/Station_3_Grid/'

def main():
    dem_tif = rasterio.open(RECON_DIR + 'tiffs/dem_.tif')
    print(dem_tif.crs)

    shape_files = glob.glob(RECON_DIR + 'gpkg_files/*.gpkg')
    print(shape_files)
    for sf in shape_files:
        if any(k in sf for k in ['labelled', 'proc', 'ref', 'Pred']):
            continue
        fn = sf.split('/')[-1].split('.')[0]
        print(fn)
        gdf = gpd.read_file(sf)
        print(gdf.crs)
        new_labels_2D = []
        new_labels_3D = []
        new_geom_2D = []
        new_geom_3D = []
        for name, linestring in zip(gdf.NAME, gdf.geometry):
            line_pnts = np.array(linestring.coords)
            elavation_vals = np.array(list(dem_tif.sample(line_pnts[:, :2])))[:, 0]
            line_pnts[:, 2] = elavation_vals
            length_2D = 0
            length_3D = 0
            if len(line_pnts) == 0 or len(line_pnts) > 2:
                print("Invalid line: {}".format(name))
            for i in range(len(line_pnts) - 1):
                length_2D = geo_utils.measure_chordlen(line_pnts[i, 1], line_pnts[i, 0], line_pnts[i+1, 1], line_pnts[i+1, 0])
                length_3D = math.sqrt(length_2D**2 + (line_pnts[i+1, 2] - line_pnts[i, 2])**2)
            label = '_'.join((fn + '_').split(' '))
            new_labels_3D.append(str({'label': label+'_line_ann', 'W_3D': round(length_3D, 4)}))
            new_geom_3D.append(LineString(line_pnts))
            new_labels_2D.append(str({'label': label+'_line_ann', 'W_2D': round(length_2D, 4)}))
            new_geom_2D.append(LineString(line_pnts[:, :2]))
        gdf_2D = gpd.GeoDataFrame({'geometry': new_geom_2D, 'NAME': new_labels_2D}, geometry='geometry', crs=gdf.crs)
        gdf_2D.to_file(RECON_DIR + 'gpkg_files/' + '_'.join(fn.split(' ')) + '_proc_2D' + '.gpkg')
        gdf_3D = gpd.GeoDataFrame({'geometry': new_geom_3D, 'NAME': new_labels_3D}, geometry='geometry', crs=gdf.crs)
        gdf_3D.to_file(RECON_DIR + 'gpkg_files/' + '_'.join(fn.split(' ')) + '_proc_3D' + '.gpkg')

if __name__ == '__main__':
    main()