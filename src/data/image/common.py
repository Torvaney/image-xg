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
