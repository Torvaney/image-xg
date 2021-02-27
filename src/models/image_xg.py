import fastai.vision.all as vision


def fit_model(data_loader, model_config):
    learn = vision.cnn_learner(
        data_loader,
        vision.resnet34,
        metrics=[vision.error_rate, vision.accuracy]
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
    model.export(path)


def load_model(path):
    return vision.load_learner(path)
