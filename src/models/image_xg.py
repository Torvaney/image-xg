import functools
import json

import fastai.data.transforms
import fastai.data.block
import fastai.metrics
import fastai.vision.all
import fastai.vision.augment
import fastai.vision.learner
import fastai.vision.data
import fastai.vision.models


def get_dataloader(img_dir):
    return fastai.vision.data.ImageDataLoaders.from_folder(
        img_dir,
        train='train',
        valid='test',
        bs=16,
        shuffle_train=True,
        item_tfms=fastai.vision.augment.Resize(256, method=fastai.vision.augment.ResizeMethod.Squish)
    )


@functools.lru_cache(maxsize=50000)
def _read_xg(image_filepath, xg_map):
    shot_id = image_filepath.stem
    return xg_map[str(shot_id)]


def get_xg_dataloader(img_dir, xg_map):
    datablock = fastai.data.block.DataBlock(
        blocks=(fastai.vision.data.ImageBlock, fastai.data.block.RegressionBlock),
        get_items=fastai.data.transforms.get_image_files,
        get_y=lambda x: _read_xg(x, xg_map)
    )
    return datablock.dataloaders(
        img_dir,
        train='train',
        valid='test',
        bs=16,
        shuffle_train=True,
        item_tfms=fastai.vision.augment.Resize(256, method=fastai.vision.augment.ResizeMethod.Squish)
    )


def fit_model(dataloader, model_config):
    learn = fastai.vision.all.cnn_learner(
        dataloader,
        fastai.vision.models.resnet34,
        metrics=[fastai.metrics.error_rate, fastai.metrics.accuracy]
    )

    epochs, freeze_epochs, lr = model_config
    learn.fine_tune(epochs, freeze_epochs=freeze_epochs, base_lr=lr)
    return learn


def fit_model_xg_pretrain(xg_dataloader, class_dataloader, model_config):
    xg_learn = fastai.vision.all.cnn_learner(xg_dataloader, fastai.vision.models.resnet34)
    xg_learn.fine_tune(5, freeze_epochs=5, base_lr=1e-4)

    learn = fastai.vision.all.cnn_learner(class_dataloader, xg_learn.model)
    learn.fine_tune(5, freeze_epochs=5, base_lr=1e-4)
    return learn


def predict_xg(model, img):
    _, _, probs = model.predict(img)
    xg, _ = probs
    return float(xg)


def predict_batch_xg(model, dataloader):
    probs, _ = model.get_preds(dl=dataloader, reorder=False)
    xg = probs[:, 0]
    return xg.numpy()


def save_model(model, path):
    previous_path = model.path

    model.path = path.parent
    model.export(path.name)

    model.path = previous_path


def load_model(path, **kwargs):
    return fastai.learner.load_learner(path, **kwargs)
