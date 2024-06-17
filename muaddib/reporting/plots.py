import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

plt.ioff()


def make_3d_plot(
    exp_results,
    z_metric,  # should be the metric to explore
    x_metric="name",
    y_metric="epoch",
    x_limit=None,
    y_limit=None,
    z_limit=None,
    kind="surfer",
    path_save=None,
    **kwargs
):

    x_data = exp_results[x_metric].values
    y_data = exp_results[y_metric].values
    z_data = exp_results[z_metric].values

    # Create figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot data
    if kind == "scatter":
        ax.scatter(x_data, y_data, z_data, cmap=cm.coolwarm, **kwargs)
    if kind == "trisurfer":
        ax.plot_trisurf(x_data, y_data, z_data, cmap=cm.coolwarm, **kwargs)
    if kind == "surfer":
        unique_x = np.unique(x_data)
        unique_y = np.unique(y_data)
        X, Y = np.meshgrid(unique_x, unique_y)

        # Initialize Z with NaNs
        Z = np.full(X.shape, np.nan)

        # Fill Z with the corresponding z values
        for i in range(len(z_data)):
            x_idx = np.where(unique_x == x_data[i])[0]
            y_idx = np.where(unique_y == y_data[i])[0]
            if x_idx.size > 0 and y_idx.size > 0:
                Z[y_idx, x_idx] = z_data[i]

        ax.plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, **kwargs
        )
    if kind == "waterfall":
        # TODO: this is bad
        # Create unique sets of x and y data
        unique_x = np.unique(x_data)
        unique_y = np.unique(y_data)

        # Plot data
        for x_val in unique_x:
            mask = x_data == x_val
            ax.plot([x_val] * len(unique_y), unique_y, z_data[mask], **kwargs)

        for y_val in unique_y:
            mask = y_data == y_val
            ax.plot(unique_x, [y_val] * len(unique_x), z_data[mask], **kwargs)
    # Customize plot
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_zlabel(z_metric)
    ax.set_title("3D Plot of {}".format(z_metric))
    if x_limit is not None:
        ax.set_xlim(x_limit)
    if y_limit is not None:
        ax.set_ylim(y_limit)
    if z_limit is not None:
        ax.set_zlim(z_limit)

    # Save plot
    if path_save:
        plt.savefig(path_save)


def plot_mirror_weights_ratios(exp_results, metric, path_save=None, **kwargs):
    mw_losses = [f for f in exp_results["loss"].unique() if "mw" in f]
    mw_results = exp_results[exp_results["loss"].isin(mw_losses)]

    mw_dict = {
        "mwmse": 0,
        "mwomse": 0.7,
        "mwocmse": 0.5,
        "mwoemse": 0.3,
        "mwohmse": 0.225,
        "mwosmse": -0.3,
        "mwoomse": -0.7,
    }
    mw_results["mw_ratio"] = mw_results["loss"].map(mw_dict)
    mw_results = mw_results.sort_values(
        [
            "mw_ratio",
            "epoch",
        ]
    )
    make_3d_plot(
        mw_results,
        metric,
        "mw_ratio",
        "epoch",
        path_save=path_save,
        z_limit=(-10, 100),
        **kwargs
    )
