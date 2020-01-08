from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import torch, cv2
import numpy as np

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def cv2_to_tensor(image):
    image = torch.from_numpy(image)
    image = image.float() / torch.max(image)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    return image

def show_image(src):
    cv2.imshow('window', src)
    cv2.waitKey(0)

def display_predictions(output):
    labels = output[0]['labels']
    for label in labels:
        print(COCO_INSTANCE_CATEGORY_NAMES[label])

def process_predictions(output, threshold=0.90):
    boxes = output[0]['boxes']
    labels = output[0]['labels']
    scores = output[0]['scores']
    predictions = dict.fromkeys(output[0], [])
    for idx in range(len(scores)):
        if scores[idx] >= threshold:
            predictions['boxes'].append(boxes[idx])
            predictions['labels'].append(labels[idx])
            predictions['scores'].append(scores[idx])
    return predictions

def draw_boxes(image, predictions):
    for idx in range(len(predictions['boxes'])):
        print(len(predictions['boxes']))
        box = predictions['boxes'][idx].detach()
        box = box.numpy().astype(int)
        image = cv2.rectangle(img=image, color=(255,255,255), thickness=1, pt1=(box[0],0), pt2=(100, 100))
    return image


if __name__ == "__main__":
    image = cv2.imread('example.jpg', cv2.IMREAD_COLOR)
    tensor = cv2_to_tensor(image)
    output = model(tensor)
    predictions = process_predictions(output)
    image = draw_boxes(image, predictions)
    show_image(image)
    
    
