# import the necessary package
import time
import cv2
import numpy as np
import pygame
from pygame.locals import *
import socket
import urllib2

class NeuralNetwork(object):
    def __init__(self):
        self.model = cv2.ANN_MLP()
        self.layer_sizes = np.int32([50400, 32, 3])
        self.model.create(self.layer_sizes)
        self.model.load('ann_param/ann.xml')

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)


class CollectTrainingData(object):
    def __init__(self):
        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.1.93', 8000))
        self.server_socket.listen(0)

        # accept one client's connection
        self.connection = self.server_socket.accept()[0].makefile('rb')
        self.isReceiving = True

        # set up ANN
        self.ann = NeuralNetwork()

        # pygame.key.set_repeat(1, 40)
        self.collect_imgdata()

    def collect_imgdata(self):

        print 'start self-Driving ....'

        # get the video stream from our client(Pi)
        try:
            stream_bytes = ' '

            while self.isReceiving:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')

                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    gray_image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)

                    # only get the lower half image (cut the row num in half)
                    half_gray = gray_image[120:240, :]

                    # show current frame (show video in big picture)
                    cv2.imshow('view', half_gray)
                    key = cv2.waitKey(1) & 0xFF
                    # reshape half_gray from into one row numpy array
                    temp_image_array = half_gray.reshape(1, 50400).astype(np.float32)

                    prediction = self.ann.predict(temp_image_array)

                    if prediction == 0:
                        urllib2.urlopen('http://192.168.1.149:5000/forward').read()
                        print 'forward'
                    elif prediction == 1:
                        urllib2.urlopen('http://192.168.1.149:5000/left').read()
                        print 'left'
                    elif prediction == 2:
                        urllib2.urlopen('http://192.168.1.149:5000/right').read()
                        print 'right'
                    if key == ord('q'):
                        self.isReceiving = False

        finally:
            # self.server_socket.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            self.server_socket.close()
            print 'connection closed'

if __name__ == '__main__':
    CollectTrainingData()
