import abc
import base64
from typing import Optional

import cv2
import torch
import numpy as np
from loguru import logger

from iopaint.helper import (
    resize_max_size,
    pad_img_to_modulo,
)
from iopaint.schema import InpaintRequest

from iopaint.plugins.realesrgan import RealESRGANUpscaler
from iopaint.schema import RunPluginRequest


class InpaintModel:
    name = "base"
    min_size: Optional[int] = None
    pad_mod = 8
    pad_to_square = False
    is_erase_model = False

    def __init__(self, device, **kwargs):
        """

        Args:
            device:
        """
        self.device = device
        self.init_model(device, **kwargs)

    @abc.abstractmethod
    def init_model(self, device, **kwargs): ...


    @abc.abstractmethod
    def forward(self, image, mask, config: InpaintRequest):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W, 1] 255 为 masks 区域
        return: BGR IMAGE
        """
        ...

    @staticmethod
    def download(): ...

    def _pad_forward(self, image, mask, config: InpaintRequest):
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(
            image, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )
        pad_mask = pad_img_to_modulo(
            mask, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )

        # logger.info(f"final forward pad size: {pad_image.shape}")

        image, mask = self.forward_pre_process(image, mask, config)

        result = self.forward(pad_image, pad_mask, config)
        result = result[0:origin_height, 0:origin_width, :]

        result, image, mask = self.forward_post_process(result, image, mask, config)

        return result

    def forward_pre_process(self, image, mask, config):
        return image, mask

    def forward_post_process(self, result, image, mask, config):
        return result, image, mask

    @torch.no_grad()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        inpaint_result = None

        if max(image.shape) > config.hd_strategy_resize_limit:
            origin_size = image.shape[:2]
            downsize_image = resize_max_size(
                image, size_limit=config.hd_strategy_resize_limit
            )
            downsize_mask = resize_max_size(
                mask, size_limit=config.hd_strategy_resize_limit
            )

            logger.info(
                f"Run resize strategy, origin size: {image.shape} forward size: {downsize_image.shape}"
            )
            inpaint_result = self._pad_forward(
                downsize_image, downsize_mask, config
            )

            #########################
            inpaint_result_uint8 = inpaint_result.astype(np.uint8)

            # _, buffer = cv2.imencode('.png', cv2.cvtColor(inpaint_result_uint8, cv2.COLOR_RGB2BGR))
            _, buffer = cv2.imencode('.png', inpaint_result_uint8)
            base64_string = base64.b64encode(buffer).decode('utf-8')
            print("=======================================================UPSCALE WITH REAL-ESRGAN===========================================================")
            model = RealESRGANUpscaler("realesr-general-x4v3", self.device)
            inpaint_result = model.gen_image(
                inpaint_result_uint8,
                RunPluginRequest(name=RealESRGANUpscaler.name, image=base64_string, scale=4),
            )
            #########################


            # The result from RealESRGAN is BGR
            # Resize the inpainted result to the original image size
            inpaint_result = cv2.cvtColor(inpaint_result, cv2.COLOR_RGB2BGR)
            inpaint_result_resized = cv2.resize(
                inpaint_result,
                (origin_size[1], origin_size[0]),
                interpolation=cv2.INTER_CUBIC,
            ).astype(np.float32)

            # Prepare the original image (convert from RGB to BGR)
            original_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.float32)

            # Prepare the mask for alpha blending
            # Convert grayscale mask (0-255) to a 3-channel float array (0.0-1.0)
            alpha = mask.astype(np.float32) / 255.0
            alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)

            # Alpha blend: (inpainted * alpha) + (original * (1 - alpha))
            blended_bgr = (inpaint_result_resized * alpha) + (original_bgr * (1.0 - alpha))
            
            # Convert back to uint8 and assign to inpaint_result
            inpaint_result = np.clip(blended_bgr, 0, 255).astype(np.uint8)

        if inpaint_result is None:
            inpaint_result = self._pad_forward(image, mask, config)

        return inpaint_result

    def _crop_box(self, image, mask, box, config: InpaintRequest):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE, (l, r, r, b)
        """
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        img_h, img_w = image.shape[:2]

        w = box_w + config.hd_strategy_crop_margin * 2
        h = box_h + config.hd_strategy_crop_margin * 2

        _l = cx - w // 2
        _r = cx + w // 2
        _t = cy - h // 2
        _b = cy + h // 2

        l = max(_l, 0)
        r = min(_r, img_w)
        t = max(_t, 0)
        b = min(_b, img_h)

        # try to get more context when crop around image edge
        if _l < 0:
            r += abs(_l)
        if _r > img_w:
            l -= _r - img_w
        if _t < 0:
            b += abs(_t)
        if _b > img_h:
            t -= _b - img_h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]

        # logger.info(f"box size: ({box_h},{box_w}) crop size: {crop_img.shape}")

        return crop_img, crop_mask, [l, t, r, b]

    def _calculate_cdf(self, histogram):
        cdf = histogram.cumsum()
        normalized_cdf = cdf / float(cdf.max())
        return normalized_cdf

    def _calculate_lookup(self, source_cdf, reference_cdf):
        lookup_table = np.zeros(256)
        lookup_val = 0
        for source_index, source_val in enumerate(source_cdf):
            for reference_index, reference_val in enumerate(reference_cdf):
                if reference_val >= source_val:
                    lookup_val = reference_index
                    break
            lookup_table[source_index] = lookup_val
        return lookup_table

    def _match_histograms(self, source, reference, mask):
        transformed_channels = []
        if len(mask.shape) == 3:
            mask = mask[:, :, -1]

        for channel in range(source.shape[-1]):
            source_channel = source[:, :, channel]
            reference_channel = reference[:, :, channel]

            # only calculate histograms for non-masked parts
            source_histogram, _ = np.histogram(source_channel[mask == 0], 256, [0, 256])
            reference_histogram, _ = np.histogram(
                reference_channel[mask == 0], 256, [0, 256]
            )

            source_cdf = self._calculate_cdf(source_histogram)
            reference_cdf = self._calculate_cdf(reference_histogram)

            lookup = self._calculate_lookup(source_cdf, reference_cdf)

            transformed_channels.append(cv2.LUT(source_channel, lookup))

        result = cv2.merge(transformed_channels)
        result = cv2.convertScaleAbs(result)

        return result

    def _run_box(self, image, mask, box, config: InpaintRequest):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE
        """
        crop_img, crop_mask, [l, t, r, b] = self._crop_box(image, mask, box, config)

        return self._pad_forward(crop_img, crop_mask, config), [l, t, r, b]
