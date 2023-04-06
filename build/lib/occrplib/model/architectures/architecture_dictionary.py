from occrplib.model.architectures.VGG16_model import create_model_VGG16
from occrplib.model.architectures.VGG16BW_model import create_model_VGG16BW
from occrplib.model.architectures.AlexNet_model import create_model_AlexNet
from occrplib.model.architectures.AlexNetBW_model import create_model_AlexNetBW
from occrplib.model.architectures.ResNet50_model import create_model_ResNet50
from occrplib.model.architectures.ResNet50BW_model import create_model_ResNet50BW
from occrplib.model.architectures.EfficientNetB0_model import create_model_EfficientNetB0
from occrplib.model.architectures.EfficientNetB0BW_model import create_model_EfficientNetB0BW
from occrplib.model.architectures.EfficientNetB4_model import create_model_EfficientNetB4
from occrplib.model.architectures.EfficientNetB4BW_model import create_model_EfficientNetB4BW
from occrplib.model.architectures.EfficientNetB7_model import create_model_EfficientNetB7
from occrplib.model.architectures.EfficientNetB7BW_model import create_model_EfficientNetB7BW

MODELS: dict = {
    "VGG16": create_model_VGG16,
    "VGG16BW": create_model_VGG16BW,
    "AlexNet": create_model_AlexNet,
    "AlexNetBW": create_model_AlexNetBW,
    "ResNet50": create_model_ResNet50,
    "ResNet50BW": create_model_ResNet50BW,
    "EfficientNetB0": create_model_EfficientNetB0,
    "EfficientNetB0BW": create_model_EfficientNetB0BW,
    "EfficientNetB4": create_model_EfficientNetB4,
    "EfficientNetB4BW": create_model_EfficientNetB4BW,
    "EfficientNetB7": create_model_EfficientNetB7,
    "EffcientNetB7BW": create_model_EfficientNetB7BW,
}
