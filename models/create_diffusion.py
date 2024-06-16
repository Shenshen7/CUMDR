
from diffusion_models.respace import SpacedDiffusion, space_timesteps
import diffusion_models.gaussian_diffusion as gd

def create_gaussian_diffusion(diffusion_steps=1000, noise_schedule='cosine', sigma_small=True):
    # default params
    steps = diffusion_steps
    learn_sigma = False
    sigma_small = False
    predict_xstart = True
    rescale_timesteps = False
    rescale_learned_sigmas = False
    timestep_respacing = ""
    scale_beta = 1.

    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    loss_type = gd.LossType.KL

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
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps
    )