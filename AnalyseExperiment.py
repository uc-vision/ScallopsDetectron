import pathlib
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
import itertools

#TODO: plot training graphs with labels on same graph, handle multiple training runs on same settings
#TODO: get average mAP values from different validation sets, plot results

OUTPUT_PARENT_DIR = '/local/ScallopMaskRCNNOutputs/'
output_sub_folders = list(pathlib.Path(OUTPUT_PARENT_DIR).iterdir())
print(output_sub_folders)

def load_json_arr(json_path):
    lines = []
    try:
        with open(json_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
    except:
        print("Cannot find json file: {}".format(json_path))
    return lines

print(len(output_sub_folders))
fig, axs = plt.subplots(2, len(output_sub_folders), figsize=(30.0, 10.0), sharey='row', sharex='all')
if axs.ndim == 1:
    axs = axs[:, None]
for ax in axs[0]:
    ax.set(xlabel="Iteration", ylabel="mAP [0.5:0.95]")
for ax in axs[1]:
    ax.set(xlabel="Iteration", ylabel="Loss")

row_titles = ["Training Loss Components", "Mean Training & Validation Losses"]
MAX_ITT = 10000
for idx, sub_folder in enumerate(output_sub_folders):
    experiment_metrics = load_json_arr(str(sub_folder) + '/metrics.json')
    #print(experiment_metrics)
    valid_keys = sum([[key for key in x.keys() if 'mAP' in key] for x in experiment_metrics], [])
    unique_loss_keys = []
    for key in valid_keys:
        if not key in unique_loss_keys:
            unique_loss_keys.append(key)
    legend_l = []
    for loss_key in unique_loss_keys:
        legend_l.append(loss_key.split('/')[-1])
        itts = np.array([x['iteration'] for x in experiment_metrics if loss_key in x])
        itts = itts[itts < MAX_ITT]
        axs[0, idx].plot(itts, [x[loss_key] for x in experiment_metrics if loss_key in x][:len(itts)])
    axs[0, idx].set_title(sub_folder.name)
    axs[0, idx].legend(legend_l, loc='best')
    axs[0, idx].grid(True)

    # fig.canvas.draw()
    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # cv2.imshow("plot", data)
    # cv2.waitKey()

    itteration_n = np.array([x['iteration'] for x in experiment_metrics if 'total_loss' in x][:MAX_ITT])
    itteration_n = itteration_n[itteration_n < MAX_ITT]
    max_idx = len(itteration_n)

    axs[1, idx].plot(itteration_n, [x['total_loss'] for x in experiment_metrics if 'total_loss' in x][:max_idx])
    axs[1, idx].plot(itteration_n, [x['loss_box_reg'] for x in experiment_metrics if 'loss_box_reg' in x][:max_idx])
    axs[1, idx].plot(itteration_n, [x['loss_cls'] for x in experiment_metrics if 'loss_cls' in x][:max_idx])
    axs[1, idx].plot(itteration_n, [x['loss_mask'] for x in experiment_metrics if 'loss_mask' in x][:max_idx])
    axs[1, idx].plot(itteration_n, [x['loss_rpn_cls'] for x in experiment_metrics if 'loss_rpn_cls' in x][:max_idx])
    axs[1, idx].plot(itteration_n, [x['loss_rpn_loc'] for x in experiment_metrics if 'loss_rpn_loc' in x][:max_idx])

    axs[1, idx].set_title(sub_folder.name)
    if idx == 0:
        axs[1, idx].legend(['total_loss', 'loss_box_reg', 'loss_cls', 'loss_mask', 'loss_rpn_cls', 'loss_rpn_loc'], loc='upper left')
    axs[1, idx].grid(True)

    eval_metrics_path = list(sub_folder.glob('*_eval_results.json'))
    if len(eval_metrics_path):
        with open(eval_metrics_path[0], 'r') as results_file:
            results_dict = results_file.read()
        print("{} results: {}".format(eval_metrics_path[0].name, results_dict))
plt.show()
