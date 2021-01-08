# -*- coding: utf-8 -*-
import click
import logging
import matplotlib
import json
import tqdm
from mplsoccer.pitch import Pitch
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def is_data_file(f):
    return Path(f).suffix == '.json'


def image_filepath(shot, output_filepath, train=True):
    """ Get the image file path from the shot (meta)data. """
    train_subdir = 'train' if train else 'test'
    goal_subdir = 'goal' if shot['shot']['outcome']['name'] == 'Goal' else 'no-goal'

    shot_dir = Path(output_filepath)/train_subdir/goal_subdir
    shot_dir.mkdir(parents=True, exist_ok=True)

    filename = f'{shot["id"]}.png'
    return shot_dir/filename


def unzip(xs):
    return zip(*xs)


def init_pitch():
    pitch = Pitch(pitch_color=None, line_color='lightgray', stripe=False)
    fig, ax = pitch.draw()
    return fig, ax


def is_gk(player):
    return player['position']['name'] == 'Goalkeeper'


def shot_marker(shot):
    body_part = shot['shot']['body_part']['name']
    if body_part == 'Right Foot':
        return matplotlib.markers.CARETUP
    if body_part == 'Left Foot':
        return matplotlib.markers.CARETDOWN
    if body_part == 'Head':
        return 'P'
    return 'P'


def extract_xy(freeze_frame, condition=lambda x: True):
    xy = [p['location'] for p in freeze_frame if condition(p)]
    if len(xy) == 0:
        return [], []
    return unzip(xy)


def create_image(shot):
    freeze_frame = shot['shot']['freeze_frame']

    fig, ax = init_pitch()

    # Add the teammates
    x, y = extract_xy(freeze_frame, lambda x: x['teammate'])
    ax.scatter(x, y, color='red')

    # Add the outfield opposition
    x, y = extract_xy(freeze_frame, lambda x: not x['teammate'] and not is_gk(x))
    ax.scatter(x, y, color='blue')

    # Add the goalkeeper
    x, y = extract_xy(freeze_frame, lambda x: not x['teammate'] and is_gk(x))
    ax.scatter(x, y, color='green')

    # Add the shooter/ball/shot location
    x, y, *__ = shot['location']
    ax.scatter(x, y, color='hotpink', marker=shot_marker(shot))

    return fig, ax


def save_image(fig, filepath):
    fig.savefig(filepath, dpi=280)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw json (one shot per file)
        from (../raw) into images ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making image dataset from raw data')

    for filepath in tqdm.tqdm(list(Path(input_filepath).iterdir())):
        if not is_data_file(filepath):
            continue

        with open(filepath, 'r') as shot_file:
            shot = json.load(shot_file)

        # TODO: Add --refresh argument to regenerate all images
        filepath = image_filepath(shot, output_filepath, train=True)
        if filepath.exists():
            continue

        if shot['shot']['type']['name'] == 'Penalty':
            logger.warning(f'Skipping event {shot["id"]} (penalty)')
            continue

        fig, ax = create_image(shot)

        # TODO: assign shots to train/test sets at random
        save_image(fig, image_filepath(shot, output_filepath, train=True))
        matplotlib.pyplot.close(fig)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
