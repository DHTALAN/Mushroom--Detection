import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay

def main():
   st.title("Binary Classification Web app")
   st.sidebar.title('Binary Classification web app')
   st.markdown("Are your mushroom edible or poisnous 🍄")
   st.sidebar.markdown("Are your mushroom edible or poisnous 🍄")


if __name__ == '__main__':
        main()

@st.cache_data
def load_data():
    data = pd.read_csv("Mus_Data.csv")
    label=LabelEncoder()
    for col in data.columns:
        data[col]=label.fit_transform(data[col])
    return data


@st.cache_data
def split(df):
    y=df.type
    x=df.drop(columns=['type'])
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    return x_train, x_test, y_train, y_test



df = load_data()

if st.sidebar.checkbox("Show raw data",False):
    st.subheader("Mushroom data set classification")
    st.write(df)

x_train,x_test,y_train,y_test=split(df)

def visualise(metrics_list):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
        disp.plot(ax=ax)
        st.pyplot(fig)
    if 'Roc curve' in metrics_list:
        st.subheader("Roc curve")
        fig, ax = plt.subplots()
        disp = RocCurveDisplay.from_estimator(model,x_test,y_test)
        disp.plot(ax=ax)
        st.pyplot(fig)
    if 'Precision-Recall' in metrics_list:
        st.subheader("Precision-Recall")
        fig, ax = plt.subplots()
        disp=PrecisionRecallDisplay.from_estimator(model,x_test,y_test)
        disp.plot(ax=ax)
        st.pyplot(fig)


class_names=['edible','poisnous']
st.sidebar.subheader('Choose the classifier')
classifier=st.sidebar.selectbox('Classifier',('Support Vector Machine(SVM)','Logistic Regression','Random Forest'))


if classifier== 'Support Vector Machine(SVM)':
    st.sidebar.subheader("Model Hyperparameters")
    C=st.sidebar.number_input("C->(Regularisation Factor)",0.01,10.0,step=0.01,key="C")
    kernel=st.sidebar.radio("Kernel",("rbf","linear"),key='kernel')
    gamma = st.sidebar.radio("Gamma->Kernel Coefficient", ("scale", "auto"), key='gamma')

    metrics=st.sidebar.multiselect("what metrics do you want to plot?",('Confusion Matrix','Roc curve','Precision-Recall'))

    if st.sidebar.button("Classify",key="Classify"):
        st.subheader("SVM->Support Vector Machine Results")
        model=SVC(C=C,kernel=kernel,gamma=gamma)
        model.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        y_pred = model.predict(x_test)
        precision=precision_score(y_test,y_pred,labels=class_names)
        recall = recall_score(y_test,y_pred, labels=class_names)
        st.write("Accuracy of model is :",accuracy)
        st.write("precision of model is :", precision)
        st.write("recall of model is :", recall)
        visualise(metrics)

if classifier== 'Logistic Regression':
    st.sidebar.subheader("Model Hyperparameters")
    C=st.sidebar.number_input("C->(Regularisation Factor)",0.01,10.0,step=0.01,key="C_LR")
    max_iter=st.sidebar.slider("Max no of iterations you want",10,510,key="max_iter")

    metrics=st.sidebar.multiselect("what metrics do you want to plot?",('Confusion Matrix','Roc curve','Precision-Recall'))

    if st.sidebar.button("Classify",key="Classify"):
        st.subheader("LR->Logistic Regression Results")
        model=LogisticRegression(C=C,max_iter=max_iter)
        model.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        y_pred = model.predict(x_test)
        precision=precision_score(y_test,y_pred,labels=class_names)
        recall = recall_score(y_test,y_pred, labels=class_names)
        st.write("Accuracy of model is :",accuracy)
        st.write("precision of model is :", precision)
        st.write("recall of model is :", recall)
        visualise(metrics)

if classifier== 'Random Forest':
    st.sidebar.subheader("Model Hyperparameters")
    no_trees=st.sidebar.number_input("No of trees you want",100,510,step=10,key="no_trees")
    max_depth=st.sidebar.number_input("Max_depth",1,7,step=1,key="max_depth")
    bootstrap=int(st.sidebar.radio("Boostraping",("1","0"),key="bootstrap"))

    metrics=st.sidebar.multiselect("what metrics do you want to plot?",('Confusion Matrix','Roc curve','Precision-Recall'))

    if st.sidebar.button("Classify",key="Classify"):
        st.subheader("RF->Random Forest")
        model=RandomForestClassifier(n_estimators=no_trees,max_depth=max_depth,bootstrap=bootstrap)
        model.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        y_pred = model.predict(x_test)
        precision=precision_score(y_test,y_pred,labels=class_names)
        recall = recall_score(y_test,y_pred, labels=class_names)
        st.write("Accuracy of model is :",accuracy)
        st.write("precision of model is :", precision)
        st.write("recall of model is :", recall)
        visualise(metrics)
