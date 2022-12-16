import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import math
import os

HOME_DIR = '/local'  # '/home/tim'
STATION_DIR = HOME_DIR + '/Dropbox/NIWA_UC/January_2021/Station_3_Grid/'

NBINS = 60

MATCH_DIST_THRESH = 0.1
ERROR_NBINS = 30

celly = lambda cell: float(cell[1]) - 1
cellx = lambda cell: float(ord(cell[0]) - 65)

R = 6378.137  # Radius of earth in KM
def convert_gpsvec_m(lat1, lon1, lat2, lon2):
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    dy = 2 * math.sin(dLat / 2)
    dx = 2 * math.cos(math.radians((lat1 + lat2) / 2)) * math.sin(dLon / 2)
    return R * np.array([dx, dy]) * 1000

STATION_NO = 3

def main():
    xls = pd.ExcelFile(HOME_DIR + '/Dropbox/NIWA_UC/January_2021/ruk2101 raw data.xlsx')
    grid_sheet = pd.read_excel(xls, 'grid')
    print(grid_sheet.keys())

    gdf = gpd.read_file(STATION_DIR + 'live_labelled_3D.gpkg')
    labelled_scallop_lengths = []
    labelled_scallop_centers = []
    for name, linestring in zip(gdf.NAME, gdf.geometry):
        line_pnts_gps = np.array(linestring.coords)
        if line_pnts_gps.shape[0] != 2:
            print("Invalid line detected!")
        scallop_center_gps = np.mean(line_pnts_gps[:2], axis=0)
        length = float(name.split('_')[-1])
        labelled_scallop_lengths.append(1000 * length)
        labelled_scallop_centers.append(scallop_center_gps)
    labelled_lengths = np.array(labelled_scallop_lengths)
    labelled_centers_gps = np.array(labelled_scallop_centers)

    gdf = gpd.read_file(STATION_DIR + 'gridref.gpkg')
    ref_datun_gps = np.array(gdf.geometry[0].coords)
    ref_vec = convert_gpsvec_m(ref_datun_gps[0, 1], ref_datun_gps[0, 0], ref_datun_gps[1, 1], ref_datun_gps[1, 0])
    ref_vec /= np.linalg.norm(ref_vec) + 1e-16
    theta = np.arctan2((ref_vec[0] - ref_vec[1]), (ref_vec[0] + ref_vec[1]))
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    labelled_centers_xy = []
    for coord_gps in labelled_centers_gps:
        xypos = convert_gpsvec_m(ref_datun_gps[0, 1], ref_datun_gps[0, 0], coord_gps[1], coord_gps[0])
        labelled_centers_xy.append(xypos)
    labelled_centers_xy = np.array(labelled_centers_xy)
    labelled_centers = np.matmul(rot, labelled_centers_xy.T).T

    lengths = np.array(grid_sheet['lgth'])[np.where(grid_sheet['station_no'] == STATION_NO)]
    diver_lengths = lengths[~np.isnan(lengths)]
    positions = np.array([[cellx(cell)+x/1000, celly(cell)+y/1000] for x, y, cell in zip(grid_sheet['x'], grid_sheet['y'], grid_sheet['gridcell'])])
    positions = positions[np.where(grid_sheet['station_no'] == STATION_NO)][~np.isnan(lengths)]

    length_errors = []
    matched_lengths_diver = []
    matched_lengths_ortho = []
    for diver_center, diver_length in zip(positions, diver_lengths):
        dists = np.linalg.norm(labelled_centers - diver_center, axis=1)
        closest_idx = np.argmin(dists)
        dist = dists[closest_idx]
        if dist < MATCH_DIST_THRESH:
            error = labelled_lengths[closest_idx] - diver_length
            if abs(error) < 30:
                length_errors.append(error)
                matched_lengths_diver.append(diver_length)
                matched_lengths_ortho.append(labelled_lengths[closest_idx])

    print("Plotting...")

    if not pathlib.Path(STATION_DIR+'plots/').exists():
        os.mkdir(STATION_DIR+'plots/')

    fig = plt.figure()
    plt.title("Scallop Size Distribution (freq. vs size [mm])")
    plt.ylabel("Frequency")
    plt.xlabel("Scallop Width [mm]")
    plt.hist(diver_lengths, bins=NBINS, alpha=0.7)
    plt.hist(labelled_scallop_lengths, bins=NBINS, alpha=0.7)
    plt.figtext(0.15, 0.76, "Total diver count: {}".format(len(diver_lengths)))
    plt.figtext(0.15, 0.73, "Total ortho count: {}".format(len(labelled_scallop_lengths)))
    plt.grid(True)
    plt.legend(['Diver measured', 'Ortho measured'])
    fig.set_size_inches(8, 6)
    plt.savefig(STATION_DIR + "plots/ScallopSizeDistOverlap.png", dpi=900)

    fig = plt.figure()
    plt.title("Scallop Size Distribution (freq. vs size [mm])")
    plt.ylabel("Frequency")
    plt.xlabel("Scallop Width [mm]")
    plt.hist([diver_lengths, labelled_scallop_lengths], bins=NBINS, alpha=0.7)
    plt.figtext(0.15, 0.76, "Total diver count: {}".format(len(diver_lengths)))
    plt.figtext(0.15, 0.73, "Total ortho count: {}".format(len(labelled_scallop_lengths)))
    plt.grid(True)
    plt.legend(['Diver measured', 'Ortho measured'])
    fig.set_size_inches(8, 6)
    plt.savefig(STATION_DIR + "plots/ScallopSizeDistSidebySide.png", dpi=900)

    fig = plt.figure()
    plt.title("Scallop Ortho Size Error [mm]")
    plt.ylabel("Frequency")
    plt.xlabel("Measurement Error [mm]")
    plt.hist(length_errors, bins=ERROR_NBINS)
    plt.figtext(0.15, 0.85, "Total matches: {}".format(len(length_errors)))
    plt.figtext(0.15, 0.82, "Avg Absolute Error: {}mm".format(round(np.abs(np.array(length_errors)).mean(), 2)))
    plt.figtext(0.15, 0.79, "Error Distribution Mean: {}mm".format(round(np.array(length_errors).mean(), 2)))
    plt.grid(True)
    fig.set_size_inches(8, 6)
    plt.savefig(STATION_DIR + "plots/ScallopSizeError_mm.png", dpi=900)

    fig = plt.figure()
    plt.title("Scallop Ortho Size Error [%]")
    plt.ylabel("Frequency")
    plt.xlabel("Measurement Error [%]")
    plt.hist(100 * np.array(length_errors) / np.array(matched_lengths_diver), bins=ERROR_NBINS)
    plt.figtext(0.15, 0.85, "Total matches: {}".format(len(length_errors)))
    plt.figtext(0.15, 0.82, "Avg Absolute Error: {}%".format(round((100 * np.abs(np.array(length_errors)) / np.array(matched_lengths_diver)).mean(), 2)))
    plt.figtext(0.15, 0.79, "Error Distribution Mean: {}%".format(round((100 * np.array(length_errors) / np.array(matched_lengths_diver)).mean(), 2)))
    plt.grid(True)
    fig.set_size_inches(8, 6)
    plt.savefig(STATION_DIR + "plots/ScallopSizeError_perc.png", dpi=900)

    fig = plt.figure()
    plt.title("Scallop Spatial Distribution")
    plt.ylabel("[m]")
    plt.xlabel("[m]")
    plt.scatter(positions[:, 0], positions[:, 1], s=10, color='blue')
    plt.scatter(labelled_centers[:, 0], labelled_centers[:, 1], s=10, color='orange')
    plt.legend(['Diver measured', 'Ortho measured'])
    plt.grid(True)
    fig.set_size_inches(8, 8)
    plt.savefig(STATION_DIR + "plots/ScallopSpatialDist.png", dpi=900)

    fig = plt.figure()
    plt.title("Scallop Ortho vs Diver Width Measurement")
    plt.ylabel("Ortho Measurement [mm]")
    plt.xlabel("Diver Measurement [mm]")
    plt.scatter(matched_lengths_diver, matched_lengths_ortho, color='red')
    plt.plot([40, 110], [40, 110], color='grey')
    plt.figtext(0.15, 0.85, "Total matches: {}".format(len(length_errors)))
    plt.grid(True)
    fig.set_size_inches(8, 8)
    plt.savefig(STATION_DIR + "plots/DiverVsOrthoErrorLine.png", dpi=900)

    plt.show()

if __name__ == '__main__':
    main()