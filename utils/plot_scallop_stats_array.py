import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import math, os

HOME_DIR = '/local'  # '/home/tim'
STATION_DIR = HOME_DIR + '/Dropbox/NIWA_UC/January_2021/Station_3_Grid/'
if not pathlib.Path(STATION_DIR + 'plots').exists():
    os.mkdir(STATION_DIR + 'plots')
if not pathlib.Path(STATION_DIR + 'plots/tagged').exists():
    os.mkdir(STATION_DIR + 'plots/tagged')

SHOW_PLOTS = True

STATION_NO = 3
ERROR_NBINS = 6

FREQ_NBINS = 15

def main():
    xls = pd.ExcelFile(HOME_DIR + '/Dropbox/NIWA_UC/January_2021/ruk2101 raw data.xlsx')
    tag_sheet = pd.read_excel(xls, 'tags')

    diver_lengths = np.array(tag_sheet['lgth'])[np.where(tag_sheet['station_no'] == STATION_NO)]
    tag_no = np.array(tag_sheet['tag'])[np.where(tag_sheet['station_no'] == STATION_NO)]
    #print(tag_no)

    # ortho_lengths_2D = [0.0846, 0.0927, 0.0978, 0.0953, 0.09, 0.0844, 0.0945, 0.0888, 0.0904, 0.0648, 0.0869, 0.0, 0.0,
    #                  0.0967, 0.0861, 0.0886, 0.0892, 0.1005, 0.0961, 0.094, 0.0805, 0.052, 0.0953, 0.085, 0.0914, 0.101,
    #                  0.0859, 0.0847, 0.0878, 0.097, 0.0998, 0.0978, 0.1046, 0.0912, 0.0924, 0.0982, 0.0493, 0.0875,
    #                  0.1028, 0.0879, 0.0951]
    ortho_lengths = [0.0853, 0.0953, 0.0982, 0.0962, 0.0902, 0.0845, 0.0948, 0.0895, 0.0909, 0.0651, 0.0879, 0.0, 0.0,
                     0.0986, 0.0862, 0.0895, 0.0895, 0.1005, 0.0974, 0.094, 0.0842, 0.0521, 0.0967, 0.0854, 0.0923, 0.101,
                     0.0867, 0.0852, 0.0878, 0.0972, 0.0998, 0.0979, 0.1057, 0.0913, 0.0926, 0.0984, 0.0504, 0.0884,
                     0.1029, 0.0883, 0.0951]
    ortho_lengths = np.array([1000 * length for length in ortho_lengths])

    print(list(zip(tag_no, ortho_lengths, diver_lengths)))

    ortho_lengths_cp = ortho_lengths.copy()
    ortho_lengths = ortho_lengths[np.where(ortho_lengths_cp > 0)]
    diver_lengths = diver_lengths[np.where(ortho_lengths_cp > 0)]

    errors = []
    valid_diver_lengths = []
    for idx in range(len(ortho_lengths)):
        err = ortho_lengths[idx] - diver_lengths[idx]
        if abs(err) < 40:
            errors.append(err)
            valid_diver_lengths.append(diver_lengths[idx])
    errors = np.array(errors)

    fig = plt.figure()
    plt.title("Tagged Scallop Size Distribution (freq. vs size [mm])")
    plt.ylabel("Frequency")
    plt.xlabel("Scallop Width [mm]")
    plt.hist(diver_lengths, bins=FREQ_NBINS, alpha=0.7)
    plt.hist(ortho_lengths, bins=FREQ_NBINS, alpha=0.7)
    plt.figtext(0.15, 0.76, "Total diver count: {}".format(len(diver_lengths)))
    plt.figtext(0.15, 0.73, "Total ortho count: {}".format(len(ortho_lengths)))
    plt.grid(True)
    plt.legend(['Diver measured', 'Ortho measured'])
    fig.set_size_inches(8, 6)
    plt.savefig(STATION_DIR + "plots/tagged/TaggedScallopSizeDistOverlap.png", dpi=900)

    fig = plt.figure()
    plt.title("Tagged Scallop Size Distribution (freq. vs size [mm])")
    plt.ylabel("Frequency")
    plt.xlabel("Scallop Width [mm]")
    plt.hist([diver_lengths, ortho_lengths], bins=FREQ_NBINS, alpha=0.7)
    plt.figtext(0.15, 0.76, "Total diver count: {}".format(len(diver_lengths)))
    plt.figtext(0.15, 0.73, "Total ortho count: {}".format(len(ortho_lengths)))
    plt.grid(True)
    plt.legend(['Diver measured', 'Ortho measured'])
    fig.set_size_inches(8, 6)
    plt.savefig(STATION_DIR + "plots/tagged/TaggedScallopSizeDistSideBySide.png", dpi=900)

    fig = plt.figure()
    plt.title("Tagged Scallop Ortho Size Error [mm]")
    plt.ylabel("Frequency")
    plt.xlabel("Measurement Error [mm]")
    plt.hist(errors, bins=ERROR_NBINS)
    plt.figtext(0.15, 0.85, "Count: {}".format(len(errors)))
    plt.figtext(0.15, 0.82, "Error STD: {}mm".format(round(np.sqrt(np.mean((errors - errors.mean())**2)), 2)))
    plt.figtext(0.15, 0.79, "Error Distribution Mean: {}mm".format(round(np.array(errors).mean(), 2)))
    plt.grid(True)
    fig.set_size_inches(8, 6)
    plt.savefig(STATION_DIR + "plots/tagged/TaggedScallopSizeError_mm.png", dpi=900)

    fig = plt.figure()
    plt.title("Tagged Scallop Ortho Size Error [%]")
    plt.ylabel("Frequency")
    plt.xlabel("Measurement Error [%]")
    plt.hist(100 * np.array(errors) / np.array(valid_diver_lengths), bins=ERROR_NBINS)
    plt.figtext(0.15, 0.85, "Count: {}".format(len(errors)))
    error_perc = 100 * errors / np.array(valid_diver_lengths)
    plt.figtext(0.15, 0.82, "Error STD: {}%".format(round(np.sqrt(np.mean((error_perc - error_perc.mean())**2)), 2)))
    plt.figtext(0.15, 0.79, "Error Distribution Mean: {}%".format(
        round((100 * np.array(errors) / np.array(valid_diver_lengths)).mean(), 2)))
    plt.grid(True)
    fig.set_size_inches(8, 6)
    plt.savefig(STATION_DIR + "plots/tagged/TaggedScallopSizeError_perc.png", dpi=900)

    fig = plt.figure()
    plt.title("Tagged Scallop Ortho vs Diver Width Measurement")
    plt.ylabel("Ortho Measurement [mm]")
    plt.xlabel("Diver Measurement [mm]")
    plt.scatter(diver_lengths, ortho_lengths, color='red')
    plt.plot([40, 110], [40, 110], color='grey')
    plt.figtext(0.15, 0.85, "Total count: {}".format(len(diver_lengths)))
    plt.grid(True)
    fig.set_size_inches(8, 8)
    plt.savefig(STATION_DIR + "plots/tagged/TaggedDiverVsOrthoErrorLine.png", dpi=900)

    if SHOW_PLOTS:
        plt.show()

if __name__ == '__main__':
    main()