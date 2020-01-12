# faster-rcnn-flask
Small flask application demonstrating the Faster R-CNN object detection model. It takes in user input images and passes it to PyTorch's
pre-trained Faster R-CNN model where it outputs predictions of objects within the input image. The predictions include bounding box
coordinates, label of the object, and the confidence score probability for that object. Using the information provided, I just mapped
the bounding boxes and labels using Open-CV and returned that image back to the user for inspection.
