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


def create_image(shot):
    """
    Create a basic freeze-frame plot with each player as a single point, and a
    wedge showing the shot angle.
    """

    freeze_frame = shot['shot']['freeze_frame']

    fig, ax = common.init_pitch()

    # Add shot "triangle" between shot and goalposts
    shot_x, shot_y, *_ = shot['location']
    post_x = 120
    post_y1, post_y2 = (36, 44)
    tri = plt.Polygon(
        [[shot_x, shot_y],
         [post_x, post_y1],
         [post_x, post_y2]],
        color='pink',
        alpha=0.5
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

    # Add the shooter/ball/shot location and metadata (body part)
    ax.scatter(shot_x, shot_y, color='hotpink', marker=shot_marker(shot))

    # Crop image to only include the attacking half
    ax.set_xlim(55, 125)

    return fig, ax
