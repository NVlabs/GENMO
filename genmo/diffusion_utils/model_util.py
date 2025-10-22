from genmo.diffusion_utils import gaussian_diffusion as gd
from genmo.diffusion_utils.respace import SpacedDiffusion, space_timesteps


def create_gaussian_diffusion(cfg, training):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.0  # no scaling
    timestep_respacing = (
        cfg.train_timestep_respacing if training else cfg.test_timestep_respacing
    )  # ''  # can be used for ddim sampling, we don't use it.
    if type(timestep_respacing) is not str:
        timestep_respacing = str(timestep_respacing)
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(cfg.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not cfg.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
