#%%
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50, deeplabv3_mobilenet_v3_large
def custom_DeepLabv3():
  model = deeplabv3_resnet101(pretrained=False, progress=True)
  model.classifier = DeepLabHead(2048, 1)

  #Set the model in training mode
  # model.train()
  return model