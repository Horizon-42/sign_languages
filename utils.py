import matplotlib.pyplot as plt
from collections.abc import Iterable
import os

def plot_dataset(dataset, columns=6, rows=3):
    fig = plt.figure(figsize=(13, 8))

    # ax enables access to manipulate each of subplots
    ax = []
    for i in range(columns * rows):
        img, label = next(dataset) if isinstance(dataset, Iterable) else dataset[i]
        if img.dim() == 4:
            img = img.squeeze(0)
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Label: " + str(label.item()))
        # rearrage the chanel order
        im = img.permute(1, 2, 0).numpy()
        # im = img.permute(2, 1, 0).numpy
        plt.imshow(im)  # show image
    plt.show()  # finally, render the plot


def get_last_dir(root_dir: str = 'runs', phase: str = 'train'):
    paths = os.listdir(root_dir)
    paths = [p for p in paths if phase in p]
    paths = sorted(paths)
    return os.path.join(root_dir, paths[-1])


def get_next_dir(root_dir: str = 'runs', phase: str = 'train'):
    paths = os.listdir(root_dir)
    paths = [p for p in paths if phase in p]
    paths = sorted(paths)

    return os.path.join(root_dir, paths[-1][:-1] + str(len(paths)))
