import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import math

STATION_NO = 3
ERROR_NBINS = 6

FREQ_NBINS = 15

def main():
    xls = pd.ExcelFile('/home/tim/Dropbox/NIWA_UC/January_2021/ruk2101 raw data.xlsx')
    tag_sheet = pd.read_excel(xls, 'tags')

    diver_lengths = np.array(tag_sheet['lgth'])[np.where(tag_sheet['station_no'] == STATION_NO)]
    tag_no = np.array(tag_sheet['tag'])[np.where(tag_sheet['station_no'] == STATION_NO)]
    #print(tag_no)

    ortho_lengths = [0.0846, 0.0927, 0.0978, 0.0953, 0.09, 0.0844, 0.0945, 0.0888, 0.0904, 0.0648, 0.0869, 0.0, 0.0,
                     0.0967, 0.0861, 0.0886, 0.0892, 0.1005, 0.0961, 0.094, 0.0805, 0.052, 0.0953, 0.085, 0.0914, 0.101,
                     0.0859, 0.0847, 0.0878, 0.097, 0.0998, 0.0978, 0.1046, 0.0912, 0.0924, 0.0982, 0.0493, 0.0875,
                     0.1028, 0.0879, 0.0951]
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

    fig = plt.figure()
    plt.title("Scallop Size Distribution (freq. vs size [mm])")
    plt.ylabel("Frequency")
    plt.xlabel("Scallop Width [mm]")
    plt.hist(diver_lengths, bins=FREQ_NBINS, alpha=0.7)
    plt.hist(ortho_lengths, bins=FREQ_NBINS, alpha=0.7)
    plt.figtext(0.15, 0.76, "Total diver count: {}".format(len(diver_lengths)))
    plt.figtext(0.15, 0.73, "Total ortho count: {}".format(len(ortho_lengths)))
    plt.grid(True)
    plt.legend(['Diver measured', 'Ortho measured'])
    fig.set_size_inches(8, 6)
    # plt.savefig(RECON_DIR + "ScallopSizeDistImg_{}.jpeg".format(key), dpi=600)

    fig = plt.figure()
    plt.title("Scallop Size Distribution (freq. vs size [mm])")
    plt.ylabel("Frequency")
    plt.xlabel("Scallop Width [mm]")
    plt.hist([diver_lengths, ortho_lengths], bins=FREQ_NBINS, alpha=0.7)
    plt.figtext(0.15, 0.76, "Total diver count: {}".format(len(diver_lengths)))
    plt.figtext(0.15, 0.73, "Total ortho count: {}".format(len(ortho_lengths)))
    plt.grid(True)
    plt.legend(['Diver measured', 'Ortho measured'])
    fig.set_size_inches(8, 6)

    fig = plt.figure()
    plt.title("Scallop Ortho Size Error [mm]")
    plt.ylabel("Frequency")
    plt.xlabel("Measurement Error [mm]")
    plt.hist(errors, bins=ERROR_NBINS)
    plt.figtext(0.15, 0.85, "Count: {}".format(len(errors)))
    plt.figtext(0.15, 0.82, "Avg Absolute Error: {}mm".format(round(np.abs(np.array(errors)).mean(), 2)))
    plt.figtext(0.15, 0.79, "Error Distribution Mean: {}mm".format(round(np.array(errors).mean(), 2)))
    plt.grid(True)
    fig.set_size_inches(8, 6)
    # plt.savefig(RECON_DIR + "ScallopSizeDistImg_{}.jpeg".format(key), dpi=600)

    fig = plt.figure()
    plt.title("Scallop Ortho Size Error [%]")
    plt.ylabel("Frequency")
    plt.xlabel("Measurement Error [%]")
    plt.hist(100 * np.array(errors) / np.array(valid_diver_lengths), bins=ERROR_NBINS)
    plt.figtext(0.15, 0.85, "Count: {}".format(len(errors)))
    plt.figtext(0.15, 0.82, "Avg Absolute Error: {}%".format(
        round((100 * np.abs(np.array(errors)) / np.array(valid_diver_lengths)).mean(), 2)))
    plt.figtext(0.15, 0.79, "Error Distribution Mean: {}%".format(
        round((100 * np.array(errors) / np.array(valid_diver_lengths)).mean(), 2)))
    plt.grid(True)
    fig.set_size_inches(8, 6)
    # # plt.savefig(RECON_DIR + "ScallopSizeDistImg_{}.jpeg".format(key), dpi=600)

    fig = plt.figure()
    plt.title("Scallop Ortho vs Diver Width Measurement")
    plt.ylabel("Ortho Measurement [mm]")
    plt.xlabel("Diver Measurement [mm]")
    plt.scatter(diver_lengths, ortho_lengths, color='red')
    plt.plot([40, 110], [40, 110], color='grey')
    plt.figtext(0.15, 0.85, "Total count: {}".format(len(diver_lengths)))
    plt.grid(True)
    fig.set_size_inches(8, 8)

    plt.show()

if __name__ == '__main__':
    main()