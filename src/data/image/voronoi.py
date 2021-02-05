import matplotlib.pyplot as plt
import scipy.spatial

from src.data.image import common


def reflect_x(xy, max_x=120):
    return [[max_x + (max_x - x), y] for x, y in xy]


def reflect_y(xy, max_y=80):
    return [[x, (max_y + (max_y - y))] for x, y in xy]


def get_region_colour(player):
    if player['teammate']:
        return 'pink'
    if not player['teammate'] and not common.is_gk(player):
        return 'skyblue'
    if common.is_gk(player):
        return 'green'


def get_body_part_colour(shot):
    body_part = shot['shot']['body_part']['name']
    if body_part == 'Right Foot':
        return 'orange'
    if body_part == 'Left Foot':
        return 'red'
    if body_part == 'Head':
        return 'magenta'
    return 'magenta'


def create_image_voronoi(shot):
    freeze_frame = shot['shot']['freeze_frame']
    xy = (
        [shot['location'][0:2]] +
        common.unzip(common.extract_xy(freeze_frame))
    )

    v = scipy.spatial.Voronoi([
        *xy,
        # Create a bounded voronoi by reflecting points in each axis
        *reflect_x(xy, 121),
        *reflect_x(xy, -1),
        *reflect_y(xy, 81),
        *reflect_y(xy, -1),
    ])

    fig, ax = common.init_pitch()

    # For each region with a corresponding player in the freeze-frame
    # (i.e. not the reflected points), get the associated voronoi region
    # and plot it
    # Note that the shooter's region is not included here (hence point_region
    # starts at index 1) and is left blank (white) in the final image
    for player, region_ix in zip(freeze_frame, v.point_region[1:]):
        region = v.regions[region_ix]
        if (-1 in region) or (len(region) == 0):
            continue

        xy = [list(v.vertices[i]) for i in region]
        plt_region = plt.Polygon(
            xy,
            color=get_region_colour(player),
            alpha=0.5
        )
        fig.gca().add_patch(plt_region)

    # Add shot angle overlaid on top of the voronoi regions
    # If we don't do this, shots right on top of the goal, and shots where
    # the shooter has no space look very similar (no white voronoi region)
    shot_x, shot_y, *__ = shot['location']
    post_x = 121
    post_y1, post_y2 = (36, 44)
    tri = plt.Polygon(
        [[shot_x, shot_y], [post_x, post_y1], [post_x, post_y2]],
        color=get_body_part_colour(shot),
        alpha=0.9
    )
    fig.gca().add_patch(tri)

    # Crop image to only include the attacking half
    # ax.set_xlim(55, 125)

    # Crop image to only include the penalty box (ish)
    ax.set_xlim(85, 125)
    ax.set_ylim(16, 64)

    return fig, ax
