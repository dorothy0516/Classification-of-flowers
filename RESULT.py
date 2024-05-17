import cv2

from tensorflow.keras.models import load_model

resize = 224
label = ('daisy','lip')
image = cv2.resize(
    cv2.imread(r"D:\大学课程\人工智能\Training\daisy\153210866_03cc9f2f36.jpg"),
    (resize, resize))
image = image.astype("float") / 255.0  # 归一化

image = image.reshape((1, image.shape[0],image.shape[1], image.shape[2]))
# 加载模型
model = load_model("D:\大学课程\人工智能\saved_model.pb")
predict = model.predict(image)
i = predict.argmax(axis=1)[0]
# 展示结果
print('——————————————————————')
print('Predict result')
print(label[i], ':', max(predict[0]) * 100, '%')

