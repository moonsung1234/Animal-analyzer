
from PIL import Image
import numpy as np
import cv2
import os

class ImageHandler :
    def __init__(self, image_width=120, image_height=120) :
        self.image_width = image_width
        self.image_height = image_height

    def getMonochrome(self, image_path) :
        screen = np.array(Image.open(image_path))
        resize = cv2.resize(screen, (self.image_width, self.image_height))
        to_image = Image.fromarray(resize).convert("L") #8bit gray
        pixel_array = np.array(to_image).reshape((self.image_width, self.image_height, 1))

        #cv2.imshow("window", pixel_array)
        #cv2.waitKey(0)
        
        return pixel_array

    def getInImageList(self, dir_path) :
        files = os.listdir(dir_path)
        images_list = []

        for _file in files :
            if ".jpg" in _file or ".png" in _file :
                images_list.append(os.path.join(dir_path, _file))
     
        return images_list

"""
path = "C:/Users/muns3/OneDrive/Desktop/node-project/mini_game_server/pictures/chat_background.jpg"
Ihr = ImageHandler()
Ihr.getMonochrome(path)
"""