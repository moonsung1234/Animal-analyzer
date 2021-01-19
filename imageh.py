
from PIL import Image
import numpy as np
import cv2

class ImageHandler :
    def __init__(self, image_width=250, image_height=250) :
        self.image_width = image_width
        self.image_height = image_height

    def setMonochrome(self, image_path) :
        screen = np.array(Image.open(image_path))
        resize = cv2.resize(screen, (self.image_width, self.image_height))
        to_image = Image.fromarray(resize).convert("L") #8bit gray
        pixel_array = np.array(to_image)

        cv2.imshow("window", pixel_array)
        cv2.waitKey(0)
        
        return pixel_array


path = "C:/Users/muns3/OneDrive/Desktop/node-project/mini_game_server/pictures/quiz_background.jpg"
Ihr = ImageHandler()
Ihr.setMonochrome(path)