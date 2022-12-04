import geopandas as gpd
from shapely.geometry import Polygon

RECON_DIR = '~/Desktop/'

def main():
    # Convert shapes to 2D

    gdf = gpd.read_file(RECON_DIR+'live_scallops.gpkg')
    print(gdf)
    print(gdf.keys())
    new_labels = []
    for name, linestring in zip(gdf.NAME, gdf.geometry):
        print(name)
        print(linestring)
    gdf.NAME = new_labels
    gdf.to_file(RECON_DIR + 'live_scallops_labelled.gpkg')

if __name__ == '__main__':
    main()