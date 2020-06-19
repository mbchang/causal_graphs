from collections import namedtuple
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

AxisInfo = namedtuple('AxisInfo', ('range', 'name'))
PlotInfo = namedtuple('PlotInfo', ('fname', 'title'))
Axis3dInfo = namedtuple('Axis3dInfo', ('x', 'y', 'z'))

def plot3d(data, axis3dinfo, plot_info, extract_group=lambda x: x, extract_value=lambda x: x):
    """
        data: {
            xlabel0: {
                ylabel0: value00,
                ylabel1: value01
            }
            xlabel1: {
                ylabel0: value10,
                ylabel1: value11
            }
        }
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    z = []
    for i, (x_label, ys_for_x) in enumerate(data.items()):
        z.append([])
        for j, (y_label, value) in enumerate(extract_group(ys_for_x).items()):
            z[i].append(extract_value(value))

    z = np.array(z)
    z = np.transpose(z)
    x, y = np.meshgrid(axis3dinfo.x.range, axis3dinfo.y.range)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(axis3dinfo.x.name)
    ax.set_ylabel(axis3dinfo.y.name)
    ax.set_zlabel(axis3dinfo.z.name)
    ax.set_zlim([0, 20])  # Custom
    ax.view_init(30, 240)
    plt.title(plot_info.title)
    plt.savefig(plot_info.fname)
    plt.close()