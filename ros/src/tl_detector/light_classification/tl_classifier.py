from styx_msgs.msg import TrafficLight
import rospy
import cv2
import numpy as np
import tensorflow as tf
import os

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # Minimum score for classifier to consider a positive result
        pass
    
   
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # return TrafficLight.UNKNOWN

        light = TrafficLight.UNKNOWN
        # HSV allows count color within hue range
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        RED1 = np.array([0, 100, 100], dtype=np.uint8)
        RED2 = np.array([10, 255, 255], np.uint8)
        RED3 = np.array([160, 100, 100], np.uint8)
        RED4 = np.array([179, 255, 255], np.uint8)

        YELLOW1 = np.array([40.0/360*255, 100, 100], np.uint8)
        YELLOW2 = np.array([66.0/360*255, 255, 255], np.uint8)

        GREEN1 = np.array([40.0/360*255, 100, 100], np.uint8)
        GREEN2 = np.array([66.0/360*255, 255, 255], np.uint8)

        red_mask1 = cv2.inRange(hsv_img, RED1, RED2)
        red_mask2 = cv2.inRange(hsv_img, RED3, RED4)
        if cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2) > 30:
            print("RED!")
            light = TrafficLight.RED

        yellow_mask = cv2.inRange(hsv_img, YELLOW1, YELLOW2)
        if cv2.countNonZero(yellow_mask) > 30:
            print("YELLOW!")
            light = TrafficLight.YELLOW

        green_mask = cv2.inRange(hsv_img, GREEN1, GREEN2)
        if cv2.countNonZero(green_mask) > 30:
            light = TrafficLight.GREEN

        return light   
        
        
