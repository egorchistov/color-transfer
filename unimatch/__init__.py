import numpy as np
import torch
import torch.nn.functional as F

from .unimatch import UniMatch
from .flow_viz import flow_tensor_to_image
from .geometry import forward_backward_consistency_check


urls: dict[str, str] = {
    "mixdata": "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth",
    "kitti": "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-kitti15-25b554d7.pth",
}

# Comments: the config below is the one corresponding to the pretrained models
# Some do not change there anything, unless you want to retrain it.

default_cfg = {
    "num_scales": 2,
    "feature_channels": 128,
    "upsample_factor": 4,
    "num_head": 1,
    "ffn_dim_expansion": 4,
    "num_transformer_layers": 6,
    "reg_refine": True,
    "task": "flow",
}


class GMFlow(UniMatch):
    r"""Module, which finds optical flow between two images.

        This is based on the original code from paper "Unifying Flow, Stereo and
        Depth Estimation". See it for more details.

        Args:
            config: Dict with initialization parameters. Do not pass it, unless you know what you are doing`.
            pretrained: Download and set pretrained weights to the model. Options: 'mixdata', 'kitti'.
                        The mixdata model is trained on several mixed public datasets, which are
                        recommended for in-the-wild use cases.

        Returns:
            Dictionary with optical flow, and occlusion maps.

        Example:
            >>> img1 = torch.randint(255, (1, 3, 384, 1248))
            >>> img2 = torch.randint(255, (1, 3, 384, 1248))
            >>> gmflow = GMFlow('mixdata')
            >>> out = gmflow(img1, img2)
        """

    def __init__(self, pretrained="mixdata", config=None):
        super().__init__(**config or default_cfg)

        checkpoint = torch.hub.load_state_dict_from_url(urls[pretrained], map_location="cpu")

        super().load_state_dict(checkpoint["model"], strict=False)
        super().eval()

    def forward(self, img0, img1,
                padding_factor=32,
                inference_size=None,
                attn_type="swin",
                attn_splits_list=(2, 8),
                corr_radius_list=(-1, 4),
                prop_radius_list=(-1, 1),
                num_reg_refine=6,
                pred_bidir_flow=False,
                pred_bwd_flow=False,
                pred_flow_viz=False,
                fwd_bwd_consistency_check=False,
                **kwargs
                ):
        """
        Args:
            im0, im1: uint8 images with shapes :math:`(N, 3, H, W)`

        Returns:
            - ``flow``, optical flow from im0 to im1 :math:`(N, 2, H, W)`.
            - ``flow_bwd``, optical flow from im1 to im0 :math:`(N, 2, H, W)`.
            - ``fwd_occ``, ``fwd_occ`` occlusion masks [0, 1] :math:`(N, 1, H, W)`.
            - ``flow_viz``, ``flow_bwd_viz`` visualization for optical flow from im0 to im1 :math:`(N, 3, H, W)`
        """
        if fwd_bwd_consistency_check:
            assert pred_bidir_flow

        fixed_inference_size = inference_size
        transpose_img = False

        # the model is trained with size: width > height
        if img0.size(-2) > img0.size(-1):
            img0 = torch.transpose(img0, -2, -1)
            img1 = torch.transpose(img1, -2, -1)
            transpose_img = True

        nearest_size = [int(np.ceil(img0.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(img0.size(-1) / padding_factor)) * padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = img0.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            img0 = F.interpolate(img0, size=inference_size, mode='bilinear',
                                 align_corners=True)
            img1 = F.interpolate(img1, size=inference_size, mode='bilinear',
                                 align_corners=True)

        if pred_bwd_flow:
            img0, img1 = img1, img0

        results_dict = super().forward(img0, img1,
                                       attn_type=attn_type,
                                       attn_splits_list=attn_splits_list,
                                       corr_radius_list=corr_radius_list,
                                       prop_radius_list=prop_radius_list,
                                       num_reg_refine=num_reg_refine,
                                       task='flow',
                                       pred_bidir_flow=pred_bidir_flow,
                                       **kwargs,
                                       )

        flow_pr = results_dict['flow_preds'][-1]  # [B or 2 * B, 2, H, W]

        # resize back
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        flow = flow_pr[::2] if pred_bidir_flow else flow_pr  # [B, 2, H, W]

        results = {"flow": flow}

        if pred_flow_viz:
            flow_viz = np.stack([flow_tensor_to_image(item) for item in flow])

            results.update({"flow_viz": flow_viz})

        # also predict backward flow
        if pred_bidir_flow:
            assert flow_pr.size(0) % 2 == 0  # [2 * B, 2, H, W]
            flow_bwd = flow_pr[1::2]  # [B, 2, H, W]

            results.update({"flow_bwd": flow_bwd})

            if pred_flow_viz:
                flow_bwd_viz = np.stack([flow_tensor_to_image(item) for item in flow_bwd])

                results.update({"flow_bwd_viz": flow_bwd_viz})

            # forward-backward consistency check
            # occlusion is 1
            if fwd_bwd_consistency_check:
                fwd_occ, bwd_occ = forward_backward_consistency_check(flow, flow_bwd)  # [B, H, W] float

                results.update({"fwd_occ": fwd_occ.unsqueeze(1),
                                "bwd_occ": bwd_occ.unsqueeze(1)})

        return results
