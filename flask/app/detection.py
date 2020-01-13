from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import torch, cv2, base64
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

def add_border(image):
    border_vertical = int(image.shape[0] * 0.05)
    border_horizontal = int(image.shape[1] * 0.05)
    image = cv2.copyMakeBorder(
            src = image,
            top = border_vertical,
            bottom = border_vertical,
            left = border_horizontal,
            right = border_horizontal,
            borderType = cv2.BORDER_CONSTANT
        )
    return image

def process_predictions(output, threshold=0.70):
    boxes = output[0]['boxes']
    labels = output[0]['labels']
    scores = output[0]['scores']
    predictions = []
    for idx in range(len(scores)):
        if scores[idx] >= threshold:
            predictions.append({
                'box': boxes[idx],
                'label': COCO_INSTANCE_CATEGORY_NAMES[labels[idx]],
                'score': scores[idx]
            })
    if len(predictions) is 0:
        idx = scores.index(max(scores))
        predictions.append({
            'box': boxes[idx],
            'label': COCO_INSTANCE_CATEGORY_NAMES[labels[idx]],
            'score': scores[idx]
        })
    return predictions

def draw_boxes(image, predictions):
    size_factor = min(image.shape[:-1]) // 500 + 1
    for obj in predictions:        
        image = cv2.rectangle(
            img = image,
            pt1 = (int(obj['box'][0].round()), int(obj['box'][1].round())),
            pt2 = (int(obj['box'][2].round()), int(obj['box'][3].round())),
            color = (0, 0, 255),
            thickness = size_factor
        )
        image = cv2.putText(
            img = image,
            text = obj['label'],
            org = (obj['box'][0], obj['box'][1] - 5),
            fontFace = cv2.FONT_HERSHEY_PLAIN,
            fontScale = size_factor,
            color = (0, 0, 255),
            thickness = size_factor
        )
    return image

def decode_image_file(image_file):
    filestr = image_file.read()
    npimg = np.fromstring(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return image

def detect(image_file):
    image = decode_image_file(image_file)
    image = add_border(image)
    tensor = cv2_to_tensor(image)
    output = model(tensor)
    predictions = process_predictions(output)
    image = draw_boxes(image, predictions)
    retval, buffer = cv2.imencode('.png', image)
    data_uri = base64.b64encode(buffer).decode('ascii')
    return data_uri, predictions

if __name__ == "__main__":
    image = cv2.imread('example_input.jpg', cv2.IMREAD_COLOR)
    image = add_border(image)
    tensor = cv2_to_tensor(image)
    output = model(tensor)
    predictions = process_predictions(output)
    print(predictions)
    image = draw_boxes(image, predictions)
    show_image(image)
    
    
