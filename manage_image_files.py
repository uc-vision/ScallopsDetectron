import pathlib

paths = list(pathlib.Path('/local/ScallopReconstructions/gopro_118/left/').iterdir())
print("Num files: {}".format(len(paths)))

frame_nums = [int(str(pth).split('_')[-1][:-4]) for pth in paths]
path_tuples = list(zip(frame_nums, paths))
path_tuples.sort()

for num, pth in path_tuples:
    if num % 3 != 0:
        print(num)
        print(pth)
        pth.unlink()