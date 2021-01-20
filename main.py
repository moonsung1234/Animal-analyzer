
from analyzer import AnimalAnalyzer

#분류할 동물 목록
feature = ["dog", "cat"]

#이미지 등 을 저장할 경로
path = "C:/Users/muns3/OneDrive/Desktop/python-project/learning-program/animal_analyzer/data_related"
path2 = "C:/Users/muns3/OneDrive/Desktop/node-project/mini_game_server/pictures"

aa = AnimalAnalyzer(feature, path, loop_count=1)
#aa.getFeatureImage()
aa.learn()

#jpg_path = path + "/dog/2101192157596643.jpg"
jpg_path = path2 + "/mydog.jpg"
print(aa.predict(jpg_path))