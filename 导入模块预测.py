from sklearn.svm import  SVC
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
def main():
    iris_data=pd.read_csv(r'd:\Data\iris.csv')
    iris_predict_data = iris_data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
    test_labels=iris_data['Name'].values
    label_enc=LabelEncoder()
    y_middle=label_enc.fit_transform(test_labels)
    with open(r'd:\Data\SVM.pkl','rb') as obj:
        model=pickle.load(obj)
        obj.close()
    predict_data=model.predict(iris_predict_data)
    acc=model.score(iris_predict_data.values,y_middle)
    print('预测个数：',len(predict_data))
    print('准确率是：',acc)
    print(predict_data)
if __name__=='__main__':
    main()