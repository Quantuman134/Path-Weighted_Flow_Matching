from .transport import Transport, ModelType, WeightType, LossSpace, PathType, Sampler

def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    loss_space=None,
    train_eps=None,
    sample_eps=None,
    t_min=0.0,
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    - learn_score: set model prediction to score
    - learn_noise: set model prediction to noise
    - velocity_weighted: weight loss by velocity weight
    - likelihood_weighted: weight loss by likelihood weight
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    elif prediction == "target":
        model_type = ModelType.TARGET
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    if loss_space == "velocity":
        loss_space_type = LossSpace.VELOCITY
    elif loss_space == "target":
        loss_space_type = LossSpace.TARGET
    else:
        # default: match model type
        loss_space_type = LossSpace.TARGET if model_type == ModelType.TARGET else LossSpace.VELOCITY

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }

    path_type = path_choice[path_type]

    if (path_type in [PathType.VP]):
        train_eps_new = 1e-5 if train_eps is None else train_eps
        sample_eps_new = 1e-3 if sample_eps is None else sample_eps
        train_eps, sample_eps = train_eps_new, sample_eps_new
    elif (path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY):
        train_eps_new = 1e-3 if train_eps is None else train_eps
        sample_eps_new = 1e-3 if sample_eps is None else sample_eps
        train_eps, sample_eps = train_eps_new, sample_eps_new
    else: # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0
    
    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        loss_space=loss_space_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        t_min=t_min,
    )
    
    return state