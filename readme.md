# 无人机地块分类
## 首先使用已经分割好的地块边界信息shape对图片中的地块进行分割提取，获得地块小图。根据地块的地物类型分成几类，并将图片缩放至(224,224)大小放入网络训练和预测。
## 分类网络使用ResNet-18。
## 最后，再根据分割好的地块边界信息shape和原始拍摄图像提取地块小图，放入网络预测得到地块类型，写入地块信息shape。

### 标签：
![image](https://github.com/xyc19970716/uav_object_classification/blob/main/pic/%E7%9C%9F%E5%80%BC.PNG)
### 预测：
![image](https://github.com/xyc19970716/uav_object_classification/blob/main/pic/%E9%A2%84%E6%B5%8B.PNG)

## 增加了根据标签类别对应写入shape的代码
## 增加了根据json自定义格式转为shape的代码

### 食用步骤
## 首先使用extract_samples_and_create_patches.py,更改对应路径,运行获得分类地块数据
## 运行main.py训练分类模型
## 使用extract_samples_and_predict.py，切割图中地块预测类别写入shape。