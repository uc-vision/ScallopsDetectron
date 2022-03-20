import json
import matplotlib.pyplot as plt
import pathlib

experiment_folder = '/local/ScallopMaskRCNNOutputs/'#'./output'

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_subfolders = list(pathlib.Path(experiment_folder).iterdir())
for experiment_sub_path in experiment_subfolders:
    experiment_metrics = load_json_arr(str(experiment_sub_path) + '/metrics.json')
    #print(experiment_metrics)

    plt.figure()
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
        [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    plt.title("Train and Validation Loss: " + experiment_sub_path.name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    plt.grid(True)

    plt.figure()
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
        [x['loss_box_reg'] for x in experiment_metrics if 'loss_box_reg' in x])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
        [x['loss_cls'] for x in experiment_metrics if 'loss_cls' in x])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
        [x['loss_mask'] for x in experiment_metrics if 'loss_mask' in x])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
        [x['loss_rpn_cls'] for x in experiment_metrics if 'loss_rpn_cls' in x])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
        [x['loss_rpn_loc'] for x in experiment_metrics if 'loss_rpn_loc' in x])

    plt.title("Train Loss Components: " + experiment_sub_path.name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(['loss_box_reg', 'loss_cls', 'loss_mask', 'loss_rpn_cls', 'loss_rpn_loc'], loc='upper left')
    plt.grid(True)

plt.show()