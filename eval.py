import os
from pathlib import Path
from functools import partial

import torch
import wandb

from methods import run_nn, runner
from methods.simp import SIMP
from methods.dcmc import DCMC
from methods.linear import color_transfer_between_images as ct
from methods.linear import color_transfer_in_correlated_color_space as ct_ccs
from methods.linear import monge_kantorovitch_color_transfer as mkct
from methods.iterative import automated_color_grading as acg


if __name__ == "__main__":
    datasets = [
        Path("Artificial Dataset/Test"),
        Path("Real-World Dataset/Test")
    ]

    for image_dir in datasets:
        runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_CT.png", ct)
        runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_CTCCS.png", ct_ccs)
        runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_MKCT.png", mkct)
        runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_ACG.png", acg)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run = wandb.init()

    for image_dir, model in zip(datasets, ["1v459lhl:v0", "1v459lhl:v0"]):
        artifact = run.use_artifact(f"egorchistov/color-transfer/model-{model}", type="model")
        artifact_dir = artifact.download()
        dcmc = DCMC.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))
        dcmc.to(device)
        dcmc.eval()

        runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_DCMC.png",
               partial(run_nn, device=device, model=dcmc))

    for image_dir, model in zip(datasets, ["9rk4vdu5:v0", "3mqa2f4u:v0"]):
        artifact = run.use_artifact(f"egorchistov/color-transfer/model-{model}", type="model")
        artifact_dir = artifact.download()
        simp = SIMP.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))
        simp.to(device)
        simp.eval()

        runner(image_dir / "%04d_LD.png", image_dir / "%04d_R.png", image_dir / "%04d_SIMP.png",
               partial(run_nn, device=device, model=simp))
