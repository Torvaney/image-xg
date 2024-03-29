import matplotlib
import matplotlib.pyplot as plt

from src.data.image import common


def shot_marker(shot):
    body_part = shot['shot']['body_part']['name']
    if body_part == 'Right Foot':
        return matplotlib.markers.CARETUP
    if body_part == 'Left Foot':
        return matplotlib.markers.CARETDOWN
    if body_part == 'Head':
        return 'P'
    return 'P'


def create_image_shot_angle_only(shot):
    """
    Create a basic plot with a wedge showing the shot angle.
    """

    fig, ax = common.init_pitch()

    # Add shot "triangle" between shot and goalposts
    shot_x, shot_y, *_ = shot['location']
    post_x = 120
    post_y1, post_y2 = (36, 44)
    tri = plt.Polygon(
        [[shot_x, shot_y],
         [post_x, post_y1],
         [post_x, post_y2]],
        color=common.get_body_part_colour(shot),
        alpha=1.0
    )
    fig.gca().add_patch(tri)

    # Crop image to only include the penalty box (ish)
    ax.set_xlim(90, 125)
    ax.set_ylim(16, 64)

    return fig, ax


def create_image_opponent_bubbles(shot):
    """
    Create a plot with a wedge showing the shot angle, the goalkeeper position,
    and opponent outfield players who might be able to block an accurate shot
    """

    fig, ax = common.init_pitch()

    # Add shot "triangle" between shot and goalposts
    shot_x, shot_y, *_ = shot['location']
    post_x = 120
    post_y1, post_y2 = (36, 44)
    tri = plt.Polygon(
        [[shot_x, shot_y],
         [post_x, post_y1],
         [post_x, post_y2]],
        color=common.get_body_part_colour(shot),
        alpha=1.0
    )
    fig.gca().add_patch(tri)

    # Layer over opposition players
    x, y = common.extract_xy(shot['shot']['freeze_frame'], lambda x: not x['teammate'] and not common.is_gk(x))
    for x_i, y_i in zip(x, y):
        opp_circle = plt.Circle((x_i, y_i), 2, color='lightgray', alpha=0.9)
        ax.add_patch(opp_circle)

    # Layer over goalkeeper
    x, y = common.extract_xy(shot['shot']['freeze_frame'], lambda x: not x['teammate'] and common.is_gk(x))
    if len(x) > 0:
        gk_circle = plt.Circle((x, y), 3, color='green', alpha=1)
        ax.add_patch(gk_circle)

    # Crop image to only include the penalty box (ish)
    ax.set_xlim(90, 125)
    ax.set_ylim(16, 64)

    return fig, ax


def create_image(shot):
    """
    Create a basic freeze-frame plot with each player as a single point, and a
    wedge showing the shot angle.
    """

    freeze_frame = shot['shot']['freeze_frame']

    fig, ax = common.init_pitch()

    # Add shot "triangle" between shot location and goalposts
    # This tells us the shot location, in a visually obvious way
    # (hopefuly displaying as a triangle helps to differentiate shots where
    # the view-of-goal is obscured)
    shot_x, shot_y, *_ = shot['location']
    post_x = 120
    post_y1, post_y2 = (36, 44)
    tri = plt.Polygon(
        [[shot_x, shot_y],
         [post_x, post_y1],
         [post_x, post_y2]],
        color=common.get_body_part_colour(shot),
        alpha=1
    )
    fig.gca().add_patch(tri)

    # Add the teammates
    x, y = common.extract_xy(freeze_frame, lambda x: x['teammate'])
    ax.scatter(x, y, color='red')

    # Add the outfield opposition
    x, y = common.extract_xy(freeze_frame, lambda x: not x['teammate'] and not common.is_gk(x))
    ax.scatter(x, y, color='blue')

    # Add the goalkeeper
    x, y = common.extract_xy(freeze_frame, lambda x: not x['teammate'] and common.is_gk(x))
    ax.scatter(x, y, color='green')

    # Crop image to only include the attacking half
    ax = common.crop_to_half(ax)

    return fig, ax
