import fastai.metrics
import fastai.vision.augment
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


def fit_model(dataloader, model_config):
    learn = fastai.vision.learner.cnn_learner(
        dataloader,
        fastai.vision.models.resnet34,
        metrics=[fastai.metrics.error_rate, fastai.metrics.accuracy]
    )

    learn.fit_one_cycle(*model_config[0])
    learn.unfreeze()
    learn.fit_one_cycle(*model_config[1])

    return learn


def predict_xg(model, img):
    _, _, probs = model.predict(img)
    xg, _ = probs
    return float(xg)


def predict_batch_xg(model, dataloader):
    probs, _, _ = model.get_preds(dl=dataloader)
    xg = probs[:, 1]
    return float(xg)


def save_model(model, path):
    previous_path = model.path

    model.path = path.parent
    model.export(path.name)

    model.path = previous_path


def load_model(path, **kwargs):
    return fastai.learner.load_learner(path, **kwargs)
