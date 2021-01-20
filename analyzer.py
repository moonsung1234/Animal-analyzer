
from imaged import ImageDownloader
from imageh import ImageHandler
from imagecnn import ImageCnnModel
import os

class AnimalAnalyzer :
    def __init__(self, animal_feature, save_path, loop_count=10) :
        self.animal_feature = animal_feature
        self.save_path = save_path
        
        self.Idr = ImageDownloader(self.animal_feature, self.save_path, scroll_full_count=500, scroll_count=1000)
        self.Ihr = ImageHandler()

        self.learning_rate = 0.001
        self.loop_count = loop_count
        self.learning_number = 100
        self.inputs = self.Ihr.image_width**2
        self.outputs = len(self.animal_feature)

        self.Inn = ImageCnnModel(
            self.learning_rate, 
            self.loop_count, 
            self.learning_number, 
            self.inputs, 
            self.outputs
        )

    def getFeatureImage(self) :
        self.Idr.saveImages()

    def learn(self, loop_count=10) :
        x, t = [], []

        for f in self.animal_feature :
            full_path = os.path.join(self.save_path, f)
            paths_arr = self.Ihr.getInImageList(full_path)

            for _path in paths_arr :
                x.append(self.Ihr.getMonochrome(_path).reshape(self.Ihr.image_width**2))
                t.append([1, 0] if f == "dog" else [0, 1])

        self.Inn.learn(x, t)

    def predict(self, image_path) :
        image_array = self.Ihr.getMonochrome(image_path).reshape((1, self.Ihr.image_width**2))

        return self.Inn.predict(image_array)


"""
path = "C:/Users/muns3/OneDrive/Desktop/python-project/learning-program/animal_analyzer/data_related"
feature = ["dog", "cat"]

Idr = ImageDownloader(feature, 500, path)
Idr.saveImages()

x = []
t = []

Ihr = ImageHandler()

for f in feature :
    full_path = os.path.join(path, f)
    paths_arr = Ihr.getInImageList(full_path)

    for _path in paths_arr :
        x.append(Ihr.getMonochrome(_path).reshape(Ihr.image_width**2))
        t.append([1, 0] if f == "dog" else [0, 1])

print(t)

learning_rate = 0.001
loop_count = 100
learning_number = 10
inputs = Ihr.image_width**2
outputs = len(feature)

Inn = ImageCnnModel(learning_rate, loop_count, learning_number, inputs, outputs)
Inn.learn(x, t)
"""