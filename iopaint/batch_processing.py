from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from iopaint.model.utils import torch_gc
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest

def single_inpaint(
        image_path_str: str,
        mask_path_str: str,
    ) -> np.ndarray:
        """
        Runs inpainting on a single image and returns the result
        as a NumPy array (RGB).
        """
        try:
            image_path = Path(image_path_str)
            mask_path = Path(mask_path_str)


            inpaint_request = InpaintRequest(
                 hd_strategy="Resize",
                 hd_strategy_resize_limit=512
            )

            # 2. Load Image and Mask
            img = np.array(Image.open(image_path).convert("RGB"))
            mask_img = np.array(Image.open(mask_path).convert("L"))

            model_manager = ModelManager(name="lama", device="cpu")

            # 3. Resize Mask if dimensions don't match
            if mask_img.shape[:2] != img.shape[:2]:
                print(f"Resizing mask {mask_path.name} to image {image_path.name} size.")
                mask_img = cv2.resize(
                    mask_img,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )

            # 4. Run Inpainting (using the pre-loaded model)
            # The model returns a BGR image (OpenCV default)
            inpaint_result = model_manager(img, mask_img, inpaint_request)

            # 5. Convert color from BGR (cv2) to RGB (PIL/C#)
            inpaint_result_rgb = cv2.cvtColor(inpaint_result, cv2.COLOR_BGR2RGB)

            # 6. Clean up GPU memory
            torch_gc()

            # 7. Return the raw RGB NumPy array
            return inpaint_result_rgb

        except Exception as e:
            print(f"Error during inpaint: {e}")
            logger.exception(e)
            return None