from pathlib import Path
from functools import partial

import torch
import wandb
from kornia import image_to_tensor, tensor_to_image

from methods import runner
from methods.simp import SIMP
from methods.dcmc import DCMC
from methods.linear import color_transfer_between_images as ct
from methods.linear import color_transfer_in_correlated_color_space as ct_ccs
from methods.linear import monge_kantorovitch_color_transfer as mkct
from methods.iterative import automated_color_grading as acg


@torch.no_grad()
def run_nn(target, reference, model):
    target = image_to_tensor(target, keepdim=False).float()
    reference = image_to_tensor(reference, keepdim=False).float()

    corrected_left, _ = model(target, reference)

    return tensor_to_image(corrected_left)


if __name__ == "__main__":
    image_dir = Path("datasets/dataset/Test")

    runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_CT.png", ct)
    runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_CTCCS.png", ct_ccs)
    runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_MKCT.png", mkct)
    runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_ACG.png", acg)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run = wandb.init()

    artifact = run.use_artifact("egorchistov/color-transfer/...", type="model")
    artifact_dir = artifact.download()
    dcmc = DCMC.load_from_checkpoint(Path(artifact_dir).resolve() / "model.ckpt")
    dcmc.to(device)
    dcmc.eval()

    runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_DCMC.png", partial(run_nn, model=dcmc))

    artifact = run.use_artifact("egorchistov/color-transfer/...", type="model")
    artifact_dir = artifact.download()
    simp = SIMP.load_from_checkpoint(Path(artifact_dir).resolve() / "model.ckpt")
    simp.to(device)
    simp.eval()

    runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_SIMP.png", partial(run_nn, model=simp))
