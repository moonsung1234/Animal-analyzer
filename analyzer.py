
from imaged import ImageDownloader
from imageh import ImageHandler
from imagecnn import ImageCnnModel
import os

path = "C:/Users/muns3/OneDrive/Desktop/python-project/learning-program/animal_analyzer/data_related"
feature = ["dog", "cat"]

"""
Idr = ImageDownloader(feature, 500, path)
Idr.saveImages()
"""

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
Inn.predict(x, t)