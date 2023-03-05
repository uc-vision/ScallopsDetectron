import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import math, os
from utils import geo_utils
import glob

HOME_DIR = '/local'  # '/home/tim'
STATION_DIR = HOME_DIR + '/Dropbox/NIWA_UC/January_2021/Station_3_Grid/'
if not pathlib.Path(STATION_DIR + 'plots').exists():
    os.mkdir(STATION_DIR + 'plots')

SHOW_PLOTS = True

NORMALISE_HIST = True
NBINS = 60

MATCH_DIST_THRESH = 0.1
ERROR_NBINS = 30

celly = lambda cell: float(cell[1]) - 1
cellx = lambda cell: float(ord(cell[0]) - 65)

STATION_NO = 3

def main():
    xls = pd.ExcelFile(HOME_DIR + '/Dropbox/NIWA_UC/January_2021/ruk2101 raw data.xlsx')
    grid_sheet = pd.read_excel(xls, 'grid')
    print(grid_sheet.keys())

    #gdf = gpd.read_file(STATION_DIR + 'gpkg_files/Pred_180123_1705_3D_proc_filt.gpkg')

    shape_files = glob.glob(STATION_DIR + 'gpkg_files/*.gpkg')
    print(shape_files)
    for sf in shape_files:
        fn = sf.split('/')[-1].split('.')[0]
        if not (all(k in fn for k in ['Pred', 'filt']) or all(k in fn for k in ['Ann', 'proc_3D'])) or 'Tagged' in fn:
            continue
        if '2D' in fn:
            continue
        print(fn)
        gdf = gpd.read_file(sf)
        print(gdf.crs)
        plot_folder_path = STATION_DIR + 'plots/' + fn + '/'
        if not pathlib.Path(plot_folder_path).exists():
            os.mkdir(plot_folder_path)

        labelled_scallop_lengths = []
        labelled_scallop_centers = []
        for name, linestring in zip(gdf.NAME, gdf.geometry):
            line_pnts_gps = np.array(linestring.coords)
            if line_pnts_gps.shape[0] != 2:
                print("Invalid line detected!")
            scallop_center_gps = np.mean(line_pnts_gps[:2], axis=0)
            length = max(0.03, min(float(eval(name)['W_3D']), 0.2))
            labelled_scallop_lengths.append(1000 * length)
            labelled_scallop_centers.append(scallop_center_gps)
        labelled_lengths = np.array(labelled_scallop_lengths)
        labelled_centers_gps = np.array(labelled_scallop_centers)

        gdf = gpd.read_file(STATION_DIR + 'gpkg_files/gridref.gpkg')
        ref_datum_gps = np.array(gdf.geometry[0].coords)
        ref_vec = geo_utils.convert_gps2local(ref_datum_gps[0], ref_datum_gps[1][None])[0] #geo_utils.convert_gpsvec_m(ref_datum_gps[0, 1], ref_datum_gps[0, 0], ref_datum_gps[1, 1], ref_datum_gps[1, 0])
        ref_vec /= np.linalg.norm(ref_vec) + 1e-16
        theta = np.arctan2((ref_vec[0] - ref_vec[1]), (ref_vec[0] + ref_vec[1]))
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
        labelled_centers_xy = geo_utils.convert_gps2local(ref_datum_gps[0], labelled_centers_gps)
        labelled_centers = np.matmul(rot, labelled_centers_xy.T).T

        lengths = np.array(grid_sheet['lgth'])[np.where(grid_sheet['station_no'] == STATION_NO)]
        diver_lengths = lengths[~np.isnan(lengths)]
        positions = np.array([[cellx(cell)+x/1000, celly(cell)+y/1000] for x, y, cell in zip(grid_sheet['x'], grid_sheet['y'], grid_sheet['gridcell'])])
        positions = positions[np.where(grid_sheet['station_no'] == STATION_NO)][~np.isnan(lengths)]

        length_errors = []
        matched_lengths_diver = []
        matched_lengths_ortho = []
        for diver_center, diver_length in zip(positions, diver_lengths):
            dists = np.linalg.norm(labelled_centers[:, :2] - diver_center, axis=1)
            closest_idx = np.argmin(dists)
            dist = dists[closest_idx]
            if dist < MATCH_DIST_THRESH:
                error = labelled_lengths[closest_idx] - diver_length
                if abs(error) < 30:
                    length_errors.append(error)
                    matched_lengths_diver.append(diver_length)
                    matched_lengths_ortho.append(labelled_lengths[closest_idx])
        length_err_arr = np.array(length_errors)
        print("Plotting...")

        fig = plt.figure()
        plt.title(fn + " Size Distribution (freq. vs size [mm])")
        plt.ylabel("Frequency")
        plt.xlabel("Scallop Width [mm]")
        plt.hist(diver_lengths, bins=NBINS, alpha=0.7, density=NORMALISE_HIST)
        plt.hist(labelled_scallop_lengths, bins=NBINS, alpha=0.7, density=NORMALISE_HIST)
        plt.figtext(0.15, 0.76, "Total diver count: {}".format(len(diver_lengths)))
        plt.figtext(0.15, 0.73, "Total ortho count: {}".format(len(labelled_scallop_lengths)))
        plt.grid(True)
        plt.legend(['Diver measured', 'Ortho measured'])
        fig.set_size_inches(8, 6)
        plt.savefig(plot_folder_path + "ScallopSizeDistOverlap.png", dpi=900)

        fig = plt.figure()
        plt.title(fn + " Size Distribution (freq. vs size [mm])")
        plt.ylabel("Frequency")
        plt.xlabel("Scallop Width [mm]")
        plt.hist([diver_lengths, labelled_scallop_lengths], bins=NBINS, alpha=0.7, density=NORMALISE_HIST)
        plt.figtext(0.15, 0.76, "Total diver count: {}".format(len(diver_lengths)))
        plt.figtext(0.15, 0.73, "Total ortho count: {}".format(len(labelled_scallop_lengths)))
        plt.grid(True)
        plt.legend(['Diver measured', 'Ortho measured'])
        fig.set_size_inches(8, 6)
        plt.savefig(plot_folder_path + "ScallopSizeDistSidebySide.png", dpi=900)

        fig = plt.figure()
        plt.title(fn + " Ortho Size Error [mm]")
        plt.ylabel("Frequency")
        plt.xlabel("Measurement Error [mm]")
        plt.hist(length_errors, bins=ERROR_NBINS)
        plt.figtext(0.15, 0.85, "Total matches: {}".format(len(length_errors)))
        #plt.figtext(0.15, 0.82, "Avg Absolute Error: {}mm".format(round(np.abs(np.array(length_errors)).mean(), 2)))
        plt.figtext(0.15, 0.82, "Error STD: {}mm".format(round(np.sqrt(np.mean((length_err_arr - length_err_arr.mean())**2)), 2)))
        plt.figtext(0.15, 0.79, "Error Distribution Mean: {}mm".format(round(np.array(length_errors).mean(), 2)))
        plt.grid(True)
        fig.set_size_inches(8, 6)
        plt.savefig(plot_folder_path + "ScallopSizeError_mm.png", dpi=900)

        fig = plt.figure()
        plt.title(fn + " Ortho Size Error [%]")
        plt.ylabel("Frequency")
        plt.xlabel("Measurement Error [%]")
        plt.hist(100 * np.array(length_errors) / np.array(matched_lengths_diver), bins=ERROR_NBINS)
        plt.figtext(0.15, 0.85, "Total matches: {}".format(len(length_errors)))
        length_errs_perc = 100 * length_err_arr / np.array(matched_lengths_diver)
        plt.figtext(0.15, 0.82, "Error STD: {}%".format(round(np.sqrt(np.mean((length_errs_perc - length_errs_perc.mean())**2)), 2)))
        #plt.figtext(0.15, 0.82, "Avg Absolute Error: {}%".format(round((100 * np.abs(np.array(length_errors)) / np.array(matched_lengths_diver)).mean(), 2)))
        plt.figtext(0.15, 0.79, "Error Distribution Mean: {}%".format(round((100 * np.array(length_errors) / np.array(matched_lengths_diver)).mean(), 2)))
        plt.grid(True)
        fig.set_size_inches(8, 6)
        plt.savefig(plot_folder_path + "ScallopSizeError_perc.png", dpi=900)

        fig = plt.figure()
        plt.title(fn + " Spatial Distribution")
        plt.ylabel("[m]")
        plt.xlabel("[m]")
        plt.scatter(positions[:, 0], positions[:, 1], s=10, color='blue')
        plt.scatter(labelled_centers[:, 0], labelled_centers[:, 1], s=10, color='orange')
        plt.legend(['Diver measured', 'Ortho measured'])
        plt.grid(True)
        fig.set_size_inches(8, 8)
        plt.savefig(plot_folder_path + "ScallopSpatialDist.png", dpi=900)

        fig = plt.figure()
        plt.title(fn + " Ortho vs Diver Width Measurement")
        plt.ylabel("Ortho Measurement [mm]")
        plt.xlabel("Diver Measurement [mm]")
        plt.scatter(matched_lengths_diver, matched_lengths_ortho, color='red')
        plt.plot([40, 110], [40, 110], color='grey')
        plt.figtext(0.15, 0.85, "Total matches: {}".format(len(length_errors)))
        plt.grid(True)
        fig.set_size_inches(8, 8)
        plt.savefig(plot_folder_path + "DiverVsOrthoErrorLine.png", dpi=900)

        if SHOW_PLOTS:
            plt.show()

if __name__ == '__main__':
    main()