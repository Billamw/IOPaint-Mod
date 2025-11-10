from typing import Union
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
        mask: Union[object, str],
        lama_path: str,
        esrgan_path: str,
    ) -> np.ndarray:
        """Performs inpainting on a single image.

        This function loads an image and a mask to perform inpainting. It can
        handle a mask provided as either a file path (str) or a
        custom object containing 'Data', 'Height',
        and 'Width' attributes.

        The resulting inpainted image is returned as an RGB NumPy array.

        Args:
            image_path_str (str): The file path to the input image.
            mask (Union[object, str]):
                The mask to be used for inpainting.
                - If 'str', it's treated as a file path to a mask image.
                - If 'object', it's expected to have 'Data', 'Height',
                and 'Width' attributes to reconstruct the mask.
            lama_path (str): The file path to the LaMa model checkpoint.
            esrgan_path (str): The file path to the ESRGAN model for upscaling.

        Returns:
            np.ndarray: The inpainted image as an RGB NumPy array.
            None: Returns None if an exception occurs during processing.
        """
        try:
            image_path = Path(image_path_str)
            mask_path = ""


            inpaint_request = InpaintRequest(
                 hd_strategy="Resize",
                 hd_strategy_resize_limit=512
            )

            # 2. Load Image and Mask
            img = np.array(Image.open(image_path).convert("RGB"))
            if hasattr(mask, "Data") and hasattr(mask, "Height") and hasattr(mask, "Width"):
        
                print("C# MaskData object received. Reconstructing mask...")
                
                # 1. Get the 1D byte array
                mask_bytes = np.array(mask.Data)
                
                # 2. Reshape it into the 2D (Height, Width) image
                mask_img = mask_bytes.reshape(mask.Height, mask.Width)
                
            else:
                mask_path = Path(mask)
                # Fallback logic: Assume 'mask' is a string path
                print(f"Mask path received. Loading mask from disk: {mask}")
                mask_img = np.array(Image.open(mask_path).convert("L"))
            

            model_manager = ModelManager(name="lama", device="cpu", lama_path=lama_path)

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
            inpaint_result = model_manager(img, mask_img, esrgan_path, inpaint_request)

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