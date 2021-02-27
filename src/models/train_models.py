# -*- coding: utf-8 -*-
import click
import logging
import matplotlib
import json
import random
import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from fastai import vision
from src.models import image_xg


MODEL_CONFIG = {
    'basic': [(3, slice(1e-7, 1e-2)), (3, slice(1e-7, 1e-2))],
    'triangle': [(3, slice(1e-7, 1e-2)), (3, slice(1e-7, 1e-2))],
    'voronoi': [(3, slice(1e-7, 1e-2)), (3, slice(1e-7, 1e-2))],
    'noisy_voronoi': [(3, slice(1e-7, 1e-2)), (3, slice(1e-7, 1e-2))],
    'cropped_voronoi': [(3, slice(1e-7, 1e-2)), (3, slice(1e-7, 1e-2))],
    'minimal_voronoi': [(3, slice(1e-7, 1e-2)), (3, slice(1e-7, 1e-2))],
}


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Fits models on freeze-frame images (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    for image_type, model_config in MODEL_CONFIG.items():

        img_dir = Path(input_filepath)/image_type
        dls = vision.ImageDataLoaders.from_folder(
            img_dir,
            train='train',
            valid='test',
            bs=16,
            shuffle_train=True
        )

        logger.info(f'Fitting {image_type} model...')
        model = image_xg.fit_model(dls, model_config)

        logger.info(f'Fit complete! Saving model...')
        model_path = Path(output_filepath)/f'{image_type}.pkl'
        image_xg.save_model(model, model_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
