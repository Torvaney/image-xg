# -*- coding: utf-8 -*-
import click
import logging
import matplotlib
import json
import random
import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.image import basic, voronoi


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

        if shot['shot']['type']['name'] == 'Penalty':
            logger.warning(f'Skipping event {shot["id"]} (penalty)')
            continue

        # TODO: make test_proportion a function argument
        test_proportion = 0.2
        is_train = random.random() >= test_proportion

        image_types = {
            'basic': basic.create_image,
            'triangle': basic.create_image_shot_angle_only,
            'voronoi': voronoi.create_image_voronoi,
        }
        for image_type, image_fn in image_types.items():
            image_dir = Path(output_filepath)/image_type
            image_dir.mkdir(parents=True, exist_ok=True)

            # Skip image generation if the completed image exists already
            # TODO: Add --refresh argument to regenerate all images
            filepath = image_filepath(shot, image_dir, train=is_train)
            if filepath.exists():
                continue

            fig, ax = image_fn(shot)

            save_image(fig, filepath)
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
