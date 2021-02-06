import numpy as np
from mplsoccer.pitch import Pitch


def unzip(xs):
    return list(zip(*xs))


def init_pitch():
    pitch = Pitch(pitch_color=None, line_color='whitesmoke', stripe=False)
    fig, ax = pitch.draw()

    # SB coordinates start on the left side, so we should invert the axes
    ax.invert_yaxis()

    return fig, ax


def is_gk(player):
    return player['position']['name'] == 'Goalkeeper'


def extract_xy(freeze_frame, condition=lambda x: True):
    xy = [p['location'] for p in freeze_frame if condition(p)]
    if len(xy) == 0:
        return [], []
    return unzip(xy)


def crop_to_half(ax):
    ax.set_xlim(55, 125)
    return ax


def crop_to_penalty_box(ax):
    ax.set_xlim(85, 125)
    ax.set_ylim(16, 64)
    return ax


def crop_to_six_yard_box(ax):
    ax.set_xlim(105, 125)
    ax.set_ylim(28, 52)
    return ax


def add_noise_to_coord(x, y, loc=0, scale=1):
    x = x + np.random.normal(loc, scale)
    y = y + np.random.normal(loc, scale)
    return [x, y]


def add_noise_to_coords(shot, loc=0, scale=1):
    shot_noisy = shot.copy()

    # Apply noise to each freeze-frame player
    for p in shot_noisy['shot']['freeze_frame']:
        p['location'] = add_noise_to_coord(*p['location'])

    # Apply noise to shooter
    x, y, z = shot_noisy['location']
    x, y = add_noise_to_coord(x, y)
    shot_noisy['location'] = [x, y, z]

    return shot_noisy
