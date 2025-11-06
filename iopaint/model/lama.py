import os

import cv2
import numpy as np
import torch
from pathlib import Path

from iopaint.helper import (
    norm_img,
    load_jit_model
)
from iopaint.schema import InpaintRequest
from .base import InpaintModel

LAMA_MODEL_URL = str(Path(__file__).parent.resolve() / "../models/big-lama.pt")
LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "e3aa4aaa15225a33ec84f9f4bc47e500")


class LaMa(InpaintModel):
    name = "lama"
    pad_mod = 8
    is_erase_model = True

    def init_model(self, device, **kwargs):
        self.model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()


    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res