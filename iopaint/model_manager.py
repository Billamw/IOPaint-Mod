from typing import List, Dict

import torch
from loguru import logger
import numpy as np

from iopaint.model import models
from iopaint.model.utils import torch_gc
from iopaint.schema import InpaintRequest, ModelInfo, ModelType


class ModelManager:
    def __init__(self, name: str, device: torch.device, **kwargs):
        self.name = name
        self.device = device
        self.kwargs = kwargs
        self.available_models: Dict[str, ModelInfo] = {}

        self.model = self.init_model(name, device, **kwargs)

    @property
    def current_model(self) -> ModelInfo:
        return self.available_models[self.name]

    def init_model(self, name: str, device, **kwargs):
        logger.info(f"Loading model: {name}")

        model_info = ModelInfo(
            name="lama",
            path="lama",
            model_type=ModelType.INPAINT,
        )

        kwargs = {
            **kwargs,
            "model_info": model_info,
        }


        return models[name](device, **kwargs)

    @torch.inference_mode()
    def __call__(self, image, mask, config: InpaintRequest):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1] 255 means area to repaint
            config:

        Returns:
            BGR image
        """
        return self.model(image, mask, config).astype(np.uint8)

    def switch(self, new_name: str):
        if new_name == self.name:
            return

        old_name = self.name
        old_controlnet_method = self.controlnet_method
        self.name = new_name

        if (
            self.available_models[new_name].support_controlnet
            and self.controlnet_method
            not in self.available_models[new_name].controlnets
        ):
            self.controlnet_method = self.available_models[new_name].controlnets[0]
        try:
            # TODO: enable/disable controlnet without reload model
            del self.model
            torch_gc()

            self.model = self.init_model(
                new_name, self.device, **self.kwargs
            )
        except Exception as e:
            self.name = old_name
            self.controlnet_method = old_controlnet_method
            logger.info(f"Switch model from {old_name} to {new_name} failed, rollback")
            self.model = self.init_model(
                old_name, self.device, **self.kwargs
            )
            raise e