from enum import Enum
from typing import Optional, Literal, List

from pydantic import BaseModel, Field, model_validator


class ModelType(str, Enum):
    INPAINT = "inpaint"  # LaMa, MAT...


class ModelInfo(BaseModel):
    name: str
    path: str
    model_type: ModelType
    is_single_file_diffusers: bool = False

class Choices(str, Enum):
    @classmethod
    def values(cls):
        return [member.value for member in cls]


class RealESRGANModel(Choices):
    realesr_general_x4v3 = "realesr-general-x4v3"
    RealESRGAN_x4plus = "RealESRGAN_x4plus"
    RealESRGAN_x4plus_anime_6B = "RealESRGAN_x4plus_anime_6B"


class Device(Choices):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class PluginInfo(BaseModel):
    name: str
    support_gen_image: bool = False
    support_gen_mask: bool = False


class CV2Flag(str, Enum):
    INPAINT_NS = "INPAINT_NS"
    INPAINT_TELEA = "INPAINT_TELEA"


class InpaintRequest(BaseModel):
    image: Optional[str] = Field(None, description="base64 encoded image")
    mask: Optional[str] = Field(None, description="base64 encoded mask")


    hd_strategy: str = Field(
        "Resize",
        description="Different way to preprocess image, only used by erase models(e.g. lama/mat)",
    )
    hd_strategy_resize_limit: int = Field(
        1280, description="Resize limit for hd_strategy=RESIZE"
    )


    @model_validator(mode="after")
    def validate_field(cls, values: "InpaintRequest"):

        return values


class RunPluginRequest(BaseModel):
    name: str
    image: str = Field(..., description="base64 encoded image")
    clicks: List[List[int]] = Field(
        [], description="Clicks for interactive seg, [[x,y,0/1], [x2,y2,0/1]]"
    )
    scale: float = Field(2.0, description="Scale for upscaling")



AdjustMaskOperate = Literal["expand", "shrink", "reverse"]
class AdjustMaskRequest(BaseModel):
    mask: str = Field(
        ..., description="base64 encoded mask. 255 means area to do inpaint"
    )
    operate: AdjustMaskOperate = Field(..., description="expand/shrink/reverse")
    kernel_size: int = Field(5, description="Kernel size for expanding mask")
