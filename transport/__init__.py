from .transport import Transport, ModelType, WeightType, LossSpace, PathType, Sampler

def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    loss_space=None,
    train_eps=None,
    sample_eps=None,
    t_min=0.0,
    scale_loss=False,
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
    elif loss_space == "noise":
        loss_space_type = LossSpace.NOISE
    elif loss_space == "min_snr":
        loss_space_type = LossSpace.MIN_SNR
    elif loss_space == "constant_blend_xv":
        loss_space_type = LossSpace.CONSTANT_BLEND_XV
    elif loss_space == "linear_blend_xv":
        loss_space_type = LossSpace.LINEAR_BLEND_XV
    elif loss_space == "constant_blend_xn":
        loss_space_type = LossSpace.CONSTANT_BLEND_XN
    elif loss_space == "linear_blend_xn":
        loss_space_type = LossSpace.LINEAR_BLEND_XN
    elif loss_space == "constant_blend_vn":
        loss_space_type = LossSpace.CONSTANT_BLEND_VN
    elif loss_space == "linear_blend_vn":
        loss_space_type = LossSpace.LINEAR_BLEND_VN
    elif loss_space == "constant_blend_xv_entire":
        loss_space_type = LossSpace.CONSTANT_BLEND_XV_ENTIRE
    elif loss_space == "linear_blend_xv_entire":
        loss_space_type = LossSpace.LINEAR_BLEND_XV_ENTIRE
    elif loss_space == "constant_blend_xn_entire":
        loss_space_type = LossSpace.CONSTANT_BLEND_XN_ENTIRE
    elif loss_space == "linear_blend_xn_entire":
        loss_space_type = LossSpace.LINEAR_BLEND_XN_ENTIRE
    elif loss_space == "constant_blend_vn_entire":
        loss_space_type = LossSpace.CONSTANT_BLEND_VN_ENTIRE
    elif loss_space == "linear_blend_vn_entire":
        loss_space_type = LossSpace.LINEAR_BLEND_VN_ENTIRE
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
        scale_loss=scale_loss,
    )
    
    return state