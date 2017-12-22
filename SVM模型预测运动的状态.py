import numpy as np
import pandas as pd
import pickle
from sklearn.svm import  SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
def main():
    test_data=pd.read_csv(r'D:\Data\test.csv')
    train_labels=test_data['Activity'].values
    label_enc=LabelEncoder()
    y_test=label_enc.fit_transform(train_labels)
    print('类别标签',label_enc.classes_)
    # label=['LAYING','SITTING','STANDING','WALKING','WALKING_DOWNSTAIRS','WALKING_UPSTAIRS']
    feat_names=test_data.columns[:-2].tolist()
    x_test=test_data[feat_names].values
    with open(r'd:\Data\第四课模型\SVM.pkl','rb') as obj:
        model=pickle.load(obj)
        obj.close()
    predict_data=model.predict(x_test)
    print('预测个数：',len(predict_data))
    predict_data_nn=[]
    for i in predict_data:
        predict_data_nn.append(label_enc.inverse_transform(i))
    print('预测的结果是：',predict_data_nn[:100])
    # acc=accuracy_score(y_test,predict_data)
    acc=model.score(x_test,y_test)
    print('预测准确率是',acc)

if __name__=='__main__':
    main()
