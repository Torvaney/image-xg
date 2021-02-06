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
    ax.set_xlim(90, 125)
    ax.set_ylim(16, 64)
    return ax
