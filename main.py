
from analyzer import AnimalAnalyzer

#분류할 동물 목록
feature = ["dog", "cat"]

#이미지 등 을 저장할 경로
path = "C:/Users/muns3/OneDrive/Desktop/python-project/learning-program/animal_analyzer/data_related"

aa = AnimalAnalyzer(feature, path)
aa.getFeatureImage()
aa.learn()

"""
jpg_path = path + "/cat/21011915594278978.jpg"

print(aa.predict(jpg_path))
"""