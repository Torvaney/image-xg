# -*- coding: utf-8 -*-
import click
import collections
import logging
import matplotlib
import json
import random
import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import src.data.image


def is_data_file(f):
    return Path(f).suffix == '.json'


def train_test_split_from_dict(d, test_proportion):
    """ Create a train/test split mapping from a dict and test_proportion. """
    return collections.defaultdict(lambda: random.random() >= test_proportion, d)


def is_penalty(shot):
    return shot['shot']['type']['name'] == 'Penalty'


def image_filepath(shot, output_filepath, train=True):
    """ Get the image file path from the shot (meta)data. """
    train_subdir = 'train' if train else 'test'
    goal_subdir = 'goal' if shot['shot']['outcome']['name'] == 'Goal' else 'no-goal'

    shot_dir = Path(output_filepath)/train_subdir/goal_subdir
    shot_dir.mkdir(parents=True, exist_ok=True)

    filename = f'{shot["id"]}.png'
    return shot_dir/filename


def save_image(fig, filepath):
    fig.savefig(filepath, dpi=60)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw json (one shot per file)
        from (../raw) into images ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making image dataset from raw data')

    # Create a mapping of shot ID: is_train so that we can cache whether each
    # shot was in the train or test set across different invocations of
    # `make_dataset.py`
    train_test_filepath = project_dir/'data'/'train_test_split.json'
    if train_test_filepath.exists():
        with open(train_test_filepath, 'r') as f:
            train_test_cache = json.load(f)
    else:
        logger.warning(
            f'No train/test splits found at {train_test_filepath}.'
            'Starting a new one at this location.'
        )
        train_test_cache = {}
    train_test_split = train_test_split_from_dict(
        train_test_cache,
        test_proportion=0.2  # TODO: make configurable
    )

    for filepath in tqdm.tqdm(list(Path(input_filepath).iterdir())[0:10]):
        logger.debug(f'Generating image files for {filepath}')
        if not is_data_file(filepath):
            continue

        with open(filepath, 'r') as shot_file:
            shot = json.load(shot_file)

        if is_penalty(shot):
            logger.warning(f'Skipping event {shot["id"]} (penalty)')
            continue

        is_train = train_test_split[str(shot['id'])]
        logger.debug(f'Putting {shot["id"]} into {"training" if is_train else "test"} data')

        for image_type, image_fn in src.data.image.IMAGE_TYPES.items():
            image_dir = Path(output_filepath)/image_type
            image_dir.mkdir(parents=True, exist_ok=True)

            # Skip image generation if the completed image exists already
            # in either train or test directories
            img_filepath = image_filepath(shot, image_dir, train=is_train)
            if img_filepath.exists():
                continue

            logger.debug(f'Generating {image_type} image for {shot["id"]}')
            fig, ax = image_fn(shot)

            logger.debug(f'Saving {image_type} image for {shot["id"]}')
            img_filepath = image_filepath(shot, image_dir, train=is_train)
            save_image(fig, img_filepath)
            matplotlib.pyplot.close(fig)

    logger.debug(f'Saving updated train/test splits at {train_test_filepath}')
    with open(train_test_filepath, 'w+') as f:
        json.dump(train_test_split, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
