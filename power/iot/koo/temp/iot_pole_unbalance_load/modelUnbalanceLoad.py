# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import load_model

class ModelUnbalanceLoad:
    def __init__(self):
        self.model = load_model('iot_pole_unbalance_load_model.h5')



    def predict(self, input):
        array_input = np.zeros((1,72), dtype='float')
        array_input[0] = np.round(((input - 35) / 70.0), 2)
        return self.model.predict_classes(array_input)[0][0]


