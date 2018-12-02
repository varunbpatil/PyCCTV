#!/usr/bin/env python

import os
import cv2
import keras
import shutil
import argparse
import numpy as np
from PIL import Image
from flask import Flask, Response
from multiprocessing import Process, Value


# Disable certain warnings from tensorflow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class YoloV3:
    """
    A Yolo V3 "person" detection implementation in Keras.

    Read more about the Yolo V3 model and output interpretation here:
    https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b
    https://www.kdnuggets.com/2018/05/implement-yolo-v3-object-detector-pytorch-part-1.html
    """

    def __init__(self, model):
        """
        Load the Yolo V3 model from disk.
        """
        self.yolo_v3 = keras.models.load_model(model, compile=False)
        self.input_size = (416, 416)
        self.threshold = 0.9


    def _resize_image(self, image):
        """
        Resize image to self.input_size
        """
        iw, ih = image.size
        w, h = self.input_size
        scale = min(w/iw, h/ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', self.input_size, (128, 128, 128))
        new_image.paste(image, ((w - nw)//2, (h - nh)//2))
        return new_image


    def _predict(self, image):
        """
        Make a prediction for the given image using Yolo V3.
        """

        # Resize and normalize the image.
        image = self._resize_image(image)
        image_data = np.array(image, dtype='float32')
        # Shape of image_data is now (416, 416, 3).
        image_data /= 255.
        image_data = np.expand_dims(image_data, axis=0)
        # Shape of image_data is now (1, 416, 416, 3).

        # Return the Yolo V3 prediction for the given image.
        return self.yolo_v3.predict(image_data)


    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def contains_person(self, image):
        """
        Check whether the given image contains a person using Yolo V3.
        """

        # Get the Yolo V3 predictions for the given image.
        predictions = self._predict(image)

        # Check whether a person is detected in the image with high
        # confidence in any of the three predictions made by Yolo V3 at
        # different scales.
        #
        # Yolo V3 makes predictions at 3 different scales. The predictions have
        # shape - (1, 13, 13, 255), (1, 26, 26, 255), (1, 52, 52, 255).
        # The number 255 comes from 3 * 85 i.e, 3 anchor boxes and 85 values per
        # anchor box which consist of 4 box coordinates + 1 object confidence +
        # 80 class confidences (4 + 1 + 80 = 85) in order.
        #
        # "person" is the first of the 80 classes (COCO dataset).
        obj_conf_pos = [4, 89, 174]
        person_cls_pos = [5, 90, 175]

        for pred in predictions:
            x, y = pred.shape[1:3]
            pred = pred[0]

            for i in range(x):
                for j in range(y):
                    for (obj, person) in zip(obj_conf_pos, person_cls_pos):
                        if self._sigmoid(pred[i, j, obj]) > self.threshold and \
                           self._sigmoid(pred[i, j, person]) > self.threshold:
                            return True

        return False



class PyCCTV:
    """
    A CCTV camera application with "person" detection and
    remote monitoring over Wi-Fi.
    """

    def __init__(self, model, output):
        # Cleanup the output directory.
        shutil.rmtree(output, ignore_errors=True)
        if not os.path.exists(output):
            os.makedirs(output)

        self.model = model
        self.output = output


    @staticmethod
    def _web_server(output, image_num):
        """
        Flask web server for remote monitoring of webcam over Wi-Fi.
        """
        app = Flask("PyCCTV")

        @app.route('/')
        def index():
            return "Welcome to PyCCTV!"

        def read_image_from_disk():
            disk_image_name = "image_%05d.jpg" % (image_num.value - 1,)
            disk_image_path = os.path.join(output, disk_image_name)
            if os.path.exists(disk_image_path):
                im = cv2.imread(disk_image_path)
                return cv2.imencode('.jpg', im)[1].tobytes()

        @app.route('/image.jpg')
        def generate_response():
            return Response(read_image_from_disk(), mimetype='image/jpeg')

        app.run(host='0.0.0.0')


    @staticmethod
    def _webcam(model, output, image_num):
        """
        Continuously capture frames from the webcam and detect the presence
        of a person in the frame using Yolo V3.
        """
        yolo = YoloV3(model)

        while True:
            cam = cv2.VideoCapture(0)  # ls /sys/class/video4linux

            # Hardware defaults for Lenovo T440s
            cam.set(3, 1280)           # Width
            cam.set(4, 720)            # Height
            cam.set(10, 128/255)       # Brightness (max = 255)
            cam.set(11, 32/255)        # Contrast (max = 255)
            cam.set(12, 64/100)        # Saturation (max = 100)
            cam.set(13, 0.5)           # Hue (0 = -180, 1 = +180)

            # Read a frame from the webcam.
            ret, image = cam.read()
            cam.release()

            if not ret:
                raise Exception('Camera module not operational')

            # Convert from cv2 to PIL image.
            cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)

            # If the frame contains a person, save it to disk.
            if yolo.contains_person(pil_image):
                disk_image_name = "image_%05d.jpg" % (image_num.value,)
                disk_image_path = os.path.join(output, disk_image_name)
                cv2.imwrite(disk_image_path, image)
                image_num.value += 1


    def run(self):
        """
        Run the PyCCTV application.
        """

        # Shared variable to keep track of the most recent image.
        image_num = Value('d', 1)


        # Create two processes.
        # 1. webcam     - Continuously capture frames from webcam and check for
        #                 the presence of a person in the frame.
        # 2. web_server - A Flask web server for remote monitoring over Wi-Fi.
        processes = [Process(target=self._webcam,
                             args=(self.model, self.output, image_num)),
                     Process(target=self._web_server,
                             args=(self.output, image_num))]

        for p in processes:
            p.daemon = True
            p.start()

        # Gracefully handle Ctrl-C.
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()


if __name__ == "__main__":

    # Argument parser.
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to yolo v3 model")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output image directory")
    args = vars(ap.parse_args())


    # Start the PyCCTV application.
    cctv = PyCCTV(args['model'], args['output'])
    cctv.run()
