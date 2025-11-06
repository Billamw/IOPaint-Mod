from .lama import LaMa
from .opencv2 import OpenCV2

models = {
    LaMa.name: LaMa,
    OpenCV2.name: OpenCV2,
}
