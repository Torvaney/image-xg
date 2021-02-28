import fastai


def fit_model(data_loader, model_config):
    learn = fastai.vision.learner.cnn_learner(
        data_loader,
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


def save_model(model, path):
    previous_path = model.path

    model.path = path.parent
    model.export(path.name)

    model.path = previous_path


def load_model(path):
    return fastai.learner.load_learner(path)
