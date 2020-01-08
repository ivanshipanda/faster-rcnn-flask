from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import torch

model = fasterrcnn_resnet50_fpn(pretrained=True)