import matplotlib.pyplot as plt
import scipy.spatial

from src.data.image import common


def reflect_x(xy, max_x=120):
    return [[max_x + (max_x - x), y] for x, y in xy]


def reflect_y(xy, max_y=80):
    return [[x, (max_y + (max_y - y))] for x, y in xy]


def bounded_voronoi(points, xlim=(-1, 121), ylim=(-1, 81)):
    return scipy.spatial.Voronoi([
        *points,
        # Create a bounded voronoi by reflecting points in each axis
        *reflect_x(points, xlim[0]),
        *reflect_x(points, xlim[1]),
        *reflect_y(points, ylim[0]),
        *reflect_y(points, ylim[1]),
    ])


def plot_voronoi_region(fig, voronoi, region_ix, **kwargs):
    """ Add voronoi region to a plot (in-place). """
    region = voronoi.regions[region_ix]
    if (-1 in region) or (len(region) == 0):
        return fig

    region_xy = [list(voronoi.vertices[i]) for i in region]
    plt_region = plt.Polygon(region_xy, **kwargs)
    fig.gca().add_patch(plt_region)

    return fig


def create_image_voronoi(shot):
    fig, ax = common.init_pitch()

    # Create voronoi from freeze frame and shooter position
    freeze_frame = shot['shot']['freeze_frame']
    xy = (
        [shot['location'][0:2]] +
        common.unzip(common.extract_xy(freeze_frame))
    )
    voronoi = bounded_voronoi(xy)

    # For each region with a corresponding player in the freeze-frame
    # (i.e. not the reflected points), get the associated voronoi region
    # and plot it
    # Note that the shooter's region is not included here (shooter is at index 0
    # hence point_region starts at index 1) and is left blank (white) in the
    # final image
    for player, region_ix in zip(freeze_frame, voronoi.point_region[1:]):
        fig = plot_voronoi_region(
            fig,
            voronoi,
            region_ix,
            color=common.get_player_colour(player),
            alpha=0.5
        )

    # Add shot angle overlaid on top of the voronoi regions
    # If we don't do this, shots right on top of the goal, and shots where
    # the shooter has no space look very similar (no white voronoi region)
    shot_x, shot_y, *__ = shot['location']
    post_x = 121
    post_y1, post_y2 = (36, 44)
    tri = plt.Polygon(
        [[shot_x, shot_y], [post_x, post_y1], [post_x, post_y2]],
        color=common.get_body_part_colour(shot),
        alpha=0.9
    )
    fig.gca().add_patch(tri)

    # Crop image to only include the attacking half
    # ax.set_xlim(55, 125)

    # Crop image to only include the penalty box (ish)
    ax = common.crop_to_penalty_box(ax)

    return fig, ax


def create_image_voronoi_cropped(shot):
    fig, ax = create_image_voronoi(shot)
    common.crop_to_six_yard_box(ax)
    return fig, ax


def create_image_voronoi_noisy(shot):
    """
    Create a voronoi plot after applying noise to the xy coords
    """
    shot_noisy = common.add_noise_to_coords(shot)
    return create_image_voronoi(shot_noisy)


def create_image_minimal_voronoi(shot):
    """
    Create a voronoi plot using only shooter and GK locations
    """
    fig, ax = common.init_pitch()

    # Calculate the voronoi regions
    freeze_frame = shot['shot']['freeze_frame']
    xy_shooter = shot['location'][0:2]
    xy_gk = common.unzip(common.extract_xy(
        freeze_frame, lambda x: not x['teammate'] and common.is_gk(x)
    ))
    voronoi = bounded_voronoi([xy_shooter] + xy_gk)

    # Plot the shooter's region
    fig = plot_voronoi_region(
        fig,
        voronoi,
        voronoi.point_region[0],
        color=common.get_body_part_colour(shot),
        alpha=0.5
    )

    # Plot the GK's region
    # The GK might not be in-frame, in which case, we just skip this
    if len(xy_gk) > 0:
        fig = plot_voronoi_region(
            fig,
            voronoi,
            voronoi.point_region[1],
            color='green',
            alpha=0.5
        )

    # Crop image to only include the penalty box (ish)
    ax = common.crop_to_penalty_box(ax)

    return fig, ax
