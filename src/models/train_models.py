# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.models import image_xg


MODEL_CONFIG = {
    'basic': [(5, slice(1e-6, 1e-2)), (3, slice(1e-7, 1e-6))],
    'triangle': [(5, slice(1e-6, 1e-2)), (3, slice(1e-7, 1e-6))],
    'voronoi': [(5, slice(1e-6, 1e-2)), (3, slice(1e-7, 1e-6))],
    'noisy_voronoi': [(5, slice(1e-6, 1e-2)), (3, slice(1e-7, 1e-6))],
    'minimal_voronoi': [(5, slice(1e-6, 1e-2)), (3, slice(1e-7, 1e-6))],
    # 'bubbles': [(5, slice(1e-6, 1e-2)), (3, slice(1e-7, 1e-6))],
    # 'cropped_voronoi': [(5, slice(1e-6, 1e-2)), (3, slice(1e-7, 1e-6))],
}


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Fits models on freeze-frame images (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    for image_type, model_config in MODEL_CONFIG.items():
        model_path = Path(output_filepath)/f'{image_type}.pkl'
        if model_path.exists():
            logger.info(f'Model for {image_type} already exists. Skipping!')
            continue

        img_dir = Path(input_filepath)/image_type
        dls = image_xg.get_dataloader(img_dir)

        logger.info(f'Fitting {image_type} model...')
        model = image_xg.fit_model(dls, model_config)

        logger.info('Fit complete! Saving model...')
        image_xg.save_model(model, model_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
