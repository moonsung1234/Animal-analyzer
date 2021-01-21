<h1>Animal analyzer</h1>

<br/>

- cnn 알고리즘을 사용하여 동물들을 분류하는 모델.

<br/>

``` python
from analyzer import AnimalAnalyzer

feature = ["dog", "cat"] #분류할 동물들의 이름들
path = "학습할 사진들이 들어있을 폴더경로"

aa = AnimalAnalyzer(feature, path, loop_count=1) #loop_count : 학습 횟수
aa.getFeatureImage() #셀레니움을 사용하여 학습할 사진들을 가져오는 함수. 이미 가져왔으면 뺴도됨.
aa.learn()

jpg_path = "분류할 사진경로"
print(aa.predict(jpg_path))
```
