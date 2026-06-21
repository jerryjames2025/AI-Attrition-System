# MACHINE LEARNING PART

# preparing features

train_df=train_df.drop("Employee ID",axis=1)
test_df=test_df.drop("Employee ID",axis=1)

# converting Targets

target_mapping={"Stayed":0,"Left":1}
train_df["Attrition"]=train_df["Attrition"].map(target_mapping)
test_df["Attrition"]=test_df["Attrition"].map(target_mapping)

# splitting the target and features

X_train=train_df.drop("Attrition",axis=1)
y_train=train_df["Attrition"]
X_test=test_df.drop("Attrition",axis=1)
y_test=test_df["Attrition"]

# Identifying categorical columns

cat_features=X_train.select_dtypes(include=["object","string"]).columns.tolist()
print(cat_features)

# training first model

from catboost import CatBoostClassifier
model=CatBoostClassifier(iterations=1000,depth=6,learning_rate=0.05,loss_function='Logloss',eval_metric='AUC',verbose=100)
model.fit(X_train,y_train,cat_features=cat_features)

#predicting the model

y_pred=model.predict(X_test)
y_prob=model.predict_proba(X_test)[:,1]

#evaluating the model

from sklearn.metrics import(
    accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
)
print("Accuracy :",accuracy_score(y_test,y_pred))
print("Precision :",precision_score(y_test,y_pred))
print("Recall :",recall_score(y_test,y_pred))
print("F1 score :",f1_score(y_test,y_pred))
print("ROC-AUC :",roc_auc_score(y_test,y_prob))

###########################################################################

# Phase - 2

# Feature importance

import pandas as pd                       
importance_df=pd.DataFrame({
    "Feature":X_train.columns,
    "Importance": model.feature_importances_
})
importance_df=importance_df.sort_values(by="Importance",ascending=False)
print(importance_df)

# plotting it                                # why are we doing this feature importance ? 
import matplotlib.pyplot as plt
top_features=importance_df.head(10)             # because we need to know which are the top features that are influencing the model
plt.figure(figsize=(10,6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

# checking marital status and remote work feature importance

print(pd.crosstab(
    train_df["Marital Status"],
    train_df["Attrition"],
    normalize="index"
)*100)

print(pd.crosstab(
    train_df["Remote Work"],
    train_df["Attrition"],
    normalize="index"
)*100)

# we are plotting the confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print(cm)

# we are doing error analysis next

false_positive_idx=(
    (y_test==0)&(y_pred==1)
)
false_negative_idx=((y_test==1)&(y_pred==0))
false_positive=X_test[false_positive_idx]
false_negative=X_test[false_negative_idx]
print("False Positives :",len(false_positive))
print("False Negatievs :",len(false_negative))

# calibration curve

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
prob_true,prob_pred=calibration_curve(y_test,y_prob,n_bins=10)
plt.figure(figsize=(8,6))
plt.plot(prob_pred,prob_true,marker="o",label="Model")
plt.plot([0,1],[0,1],linestyle="--",label="Perfect Calibration")
plt.xlabel("Predicted Probability")
plt.ylabel("Actual Probability")
plt.title("Calibration Curve")
plt.legend()
plt.show()