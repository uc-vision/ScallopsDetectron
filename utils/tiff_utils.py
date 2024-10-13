import tifffile
import glob
from utils import geo_utils
import numpy as np

class DEM:
    def __init__(self, tiff_dir):
        self.dem_tiff_paths = glob.glob(tiff_dir + '*-dem-*.tif')
        self.dem_res_gps = []
        ortho_lonlat = []
        self.open_dem_idx = None
        self.open_dem = None
        for tif_pth in self.dem_tiff_paths:
            ortho_tiff_obj = tifffile.TiffFile(tif_pth)
            ortho_page_0 = ortho_tiff_obj.pages[0]  # Pages are resolution pyramid
            self.dem_res_gps.append(np.array(ortho_page_0.tags['ModelPixelScaleTag'].value)[:2])
            ortho_tiepoint = ortho_page_0.tags['ModelTiepointTag'].value
            ortho_lonlat.append(np.array(ortho_tiepoint)[3:5])  # Top left
            # res_xy_m = geo_utils.convert_gps2local(ortho_lonlat, [ortho_lonlat + pix_res_gps])[0]
        self.ortho_lonlat = np.array(ortho_lonlat)

    def get_elevation_gps(self, gps_pnt):
        # lon lat
        vec_gps = gps_pnt - self.ortho_lonlat
        vec_gps[np.where(vec_gps[:, 0] < 0), :] = 100
        vec_gps[np.where(vec_gps[:, 1] > 0), :] = 100
        tiff_idx = np.argmin(np.abs(vec_gps).sum(axis=1))
        pix_idx = (np.array([1, -1]) * vec_gps[tiff_idx] / self.dem_res_gps[tiff_idx])[::-1].astype(int)
        if self.open_dem_idx is None or tiff_idx != self.open_dem_idx:
            self.open_dem = tifffile.imread(self.dem_tiff_paths[tiff_idx])
            self.open_dem_idx = tiff_idx
        elevation = self.open_dem[tuple(pix_idx)]
        return elevation

    def get_3d_polygon(self, polygon_2d):
        polygon_3d = []
        for pnt in polygon_2d:
            z_val = self.get_elevation_gps(pnt)
            # TODO: deal with bad elevation values
            polygon_3d.append([pnt[0], pnt[1], z_val])
        return np.array(polygon_3d)
