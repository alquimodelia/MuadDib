def keras_train_model(
    model_obj=None,
    epochs=None,
    datamanager=None,
    optimizer=None,
    loss=None,
    batch_size=None,
    weights=None,
    callbacks=None,
    training_args=None,
    **kwargs,
):
    training_args = training_args or {}
    X, Y, _, _ = datamanager.training_data(**training_args)
    keras.backend.clear_session()

    fit_args = {
        "epochs": epochs,
        "callbacks": callbacks,
        "batch_size": batch_size,
        "shuffle": True,
    }

    compile_args = {
        "optimizer": optimizer,
        "loss": loss,
    }

    if weights:
        if weights == "mean":
            sample_weights = np.abs(np.array(Y) - datamanager.y_mean)
        else:
            sample_weights = np.arange(len(Y)) + 1
        if isinstance(weights, str):
            if "squared" in weights.lower():
                sample_weights = sample_weights**2
            if "cubic" in weights.lower():
                sample_weights = sample_weights**3
            if "inv" in weights.lower():
                sample_weights = np.max(sample_weights) / sample_weights
            if "norm" in weights.lower():
                sample_weights = sample_weights / np.max(sample_weights)

        sam = sample_weights.reshape(Y.shape[0], 1, 1)
        sample_weights = np.broadcast_to(sam, Y.shape)
        # sample_weights = sample_weights.flatten()
        fit_args.update({"sample_weight": sample_weights})

    model_obj.compile(**compile_args)
    history_new = model_obj.fit(X, Y, **fit_args, **kwargs)
    return history_new


def statsmodel_train_model(
    model_obj=None,
    modelfilepath=None,
    **kwargs,
):
    model_fit = model_obj.fit(**training_args)
    model_fit.save(modelfilepath)

    return model_fit
