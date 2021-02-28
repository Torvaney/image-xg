# -*- coding: utf-8 -*-
import click
import fastai.vision.core
import logging
import pandas as pd
import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.models import image_xg, train_models


def event_id_from_path(img_path):
    return Path(img_path).stem


@click.command()
@click.argument('image_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(image_filepath, model_filepath, output_filepath):
    """ Generate model predictions for test images (saved in ../processed) and
    save them as csvs
    """
    logger = logging.getLogger(__name__)

    xg_values = []
    for image_type, __ in train_models.MODEL_CONFIG.items():
        model_path = Path(model_filepath)/f'{image_type}.pkl'
        model = image_xg.load_model(model_path)

        logger.info(f'Generating predictions for {image_type} model...')

        img_dir = Path(image_filepath)/image_type/'test'
        for img_path in tqdm.tqdm(list(img_dir.glob('**/*.png')):
            img = fastai.vision.core.PILImage.create(img_path)
            xg = image_xg.predict_xg(model, img)

            xg_values.append({
                'image_type': image_type,
                'img_path': img_path,
                'event_id': event_id_from_path(img_path),
                'xg': xg
            })

    logger.info('Saving predictions to csv')
    pd.DataFrame(xg_values).to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
