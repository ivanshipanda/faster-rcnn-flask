# faster-rcnn-flask
Small flask application demonstrating the Faster R-CNN object detection model. It takes in user input images and passes it to PyTorch's
pre-trained Faster R-CNN model where it outputs predictions of objects within the input image. The predictions include bounding box
coordinates, label of the object, and the confidence score probability for that object. Using the information from the predictions, I just mapped the bounding boxes and labels using Open-CV and returned that image back to the user for inspection.

## Using this repository

```
$ git clone https://github.com/rae0924/faster-rcnn-flask.git
$ cd faster-rcnn-flask
$ python3 -m venv venv
$ source venv/bin/activate // venv/Scripts/activate for windows
$ pip3 install -r requirements.txt
$ python3 flask/run.py
```
The web server is hosted at http//:localhost:5000 and is accessible through your browser.

## UI

Just upload image of choice once on the web application. Hit submit. The output of the model is shown as such: 

![](results_ui.png)
