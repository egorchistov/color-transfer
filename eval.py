import os
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
def run_nn(target, reference, device, model):
    target = image_to_tensor(target, keepdim=False).float().to(device)
    reference = image_to_tensor(reference, keepdim=False).float().to(device)

    corrected_left, _ = model(target, reference)

    return tensor_to_image(corrected_left)


if __name__ == "__main__":
    datasets = [
        Path("datasets/dataset/Test"),
        Path("datasets/Real-512x512/Test")
    ]

    for image_dir in datasets:
        runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_CT.png", ct)
        runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_CTCCS.png", ct_ccs)
        runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_MKCT.png", mkct)
        runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_ACG.png", acg)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run = wandb.init()

    for image_dir, model in zip(datasets, ["1v459lhl:v0", "23zxip9a:v0"]):
        artifact = run.use_artifact(f"egorchistov/color-transfer/model-{model}", type="model")
        artifact_dir = artifact.download()
        dcmc = DCMC.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))
        dcmc.to(device)
        dcmc.eval()

        runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_DCMC.png", partial(run_nn, device=device, model=dcmc))

    for image_dir, model in zip(datasets, ["14rto6rl:v5", "m19plwtk:v0"]):
        artifact = run.use_artifact(f"egorchistov/color-transfer/model-{model}", type="model")
        artifact_dir = artifact.download()
        simp = SIMP.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))
        simp.to(device)
        simp.eval()

        runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_SIMP.png", partial(run_nn, device=device, model=simp))
