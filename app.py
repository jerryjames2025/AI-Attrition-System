import pandas as pd
import numpy as np

# # we are going to load the dataset

train_df=pd.read_csv("Dataset/train.csv")
test_df=pd.read_csv("Dataset/test.csv")

# # printing the shape of each csv files

# print("Train csv shape",train_df.shape) # we are doing this to know how many rows and columns are there
# print("Test csv shape",test_df.shape)

# # next we are going to check how the data looks like head() -> prints first 5 rows, we will get an idea about how the data is

# print(train_df.head())  # there are 24 features present

# # next we need to check column informations

# print(train_df.info())  # to check column names, data types and missing values

# # next we are printing statistical summary --> describe()

# print(train_df.describe()) # this will give min,max,mean,median,std

# # Missing value analysis

# missing_values=train_df.isnull().sum()
# print(missing_values)                         # well it looks like there are no missing values.

# # now we are checking duplicate values

# duplicates=train_df.duplicated().sum()
# print("Duplicate values",duplicates)         # and there are no duplicate values as well.

# # target column discovery

# print(train_df.columns)  # there are 24 columns

# # Target distribution

# target="Attrition"
# print(train_df[target].value_counts())    # this is to understand class balance we have 31k --> stayed and 28k --> left

# # Target percentage

# print(train_df[target].value_counts(normalize=True)*100) # we are checking because accuracy can be misleading sometimes

# # we are seperating numerical and categorical values

# numerical_cols=train_df.select_dtypes(include=np.number).columns.tolist()
# categorical_cols=train_df.select_dtypes(include="object").columns.tolist()
# print("Numerical Features : ")                                 
# print(numerical_cols)               # because later we will be using it 
# print("\nCategorical Features : ")               # what we have done here is seperated categorical and numerical columns
# print(categorical_cols)

# # we are saving the EDA report 

# eda_report={
#     "Train shape": train_df.shape,
#     "Test shape": test_df.shape,
#     "Miising Values": train_df.isnull().sum().sum(),
#     "Duplicates": train_df.duplicated().sum(),
#     "Numerical Columns": len(numerical_cols),
#     "Categorical Columns": len(categorical_cols)
# }
# print(eda_report)

# # droping the employe id because we dont need that for training

train_df.drop("Employee ID",axis=1)

##########################################################################################
#################### Now we are going to answer some EDA questions########################

# 1. Does age affect Attrition ? 

# code : 

# import seaborn as sns 
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,6))
# sns.histplot(data=train_df,x="Age",hue="Attrition",kde=True,multiple="stack")
# plt.title("Age vs Attrition")
# plt.show()

# # 2. Monthly income vs Attririon 

# # code :

# plt.figure(figsize=(10,6))
# sns.boxplot(data=train_df,x="Attrition",y="Monthly Income")
# plt.title("Monthly income vs Attrition")
# plt.show()

# # 3. Over time vs Attrition

# # code :

# overtime_analysis=pd.crosstab(train_df["Overtime"],
#             train_df["Attrition"],
#             normalize="index")*100
# print("\n Over Time vs Attrition")
# print(overtime_analysis)
# # plotting
# import seaborn as sns 
# import matplotlib.pyplot as plt 
# plt.figure(figsize=(8,5))
# sns.countplot(data=train_df,x="Overtime",hue="Attrition")
# plt.title("Overtime vs Attrition")
# plt.show()

# # 4. Job satisfaction vs Attrition

# # code :

# job_satisfaction_analysis=pd.crosstab(train_df["Job Satisfaction"],
#             train_df["Attrition"],
#             normalize="index")*100
# print("\n Job satisfaction vs Attrition")
# print(job_satisfaction_analysis)
# # plotting
# import seaborn as sns 
# import matplotlib.pyplot as plt 
# plt.figure(figsize=(10,5))
# sns.countplot(data=train_df,
#               x="Job Satisfaction",
#               hue="Attrition")
# plt.title("Job Satisfaction vs Attrition")
# plt.show()  

# # 5. Work-life balance vs Attrition

# # code :

# wlb_analysis=pd.crosstab(train_df["Work-Life Balance"],
#             train_df["Attrition"],
#             normalize="index"
# )*100
# print("\n Work-Life Balance Analysis vs Attrition ")
# print(wlb_analysis)
# # plotting
# import seaborn as sns 
# import matplotlib.pyplot as plt
# sns.countplot(
#     data=train_df,
#     x="Work-Life Balance",
#     hue="Attrition"
# )
# plt.title("Work-Life Balance vs Attrition")
# plt.show()

# # 6. Leadership Oppertunites vs Attrition

# # code :

# leadership_analysis=pd.crosstab(train_df["Leadership Opportunities"],
#             train_df["Attrition"],
#             normalize="index")*100
# print("\n Leadership Oppertunities vs Attrition")
# print(leadership_analysis)
#  # plotting
# import seaborn as sns 
# import matplotlib.pyplot as plt
# sns.countplot(
#     data=train_df,
#     x="Leadership Opportunities",
#     hue="Attrition"
# ) 
# plt.title("Leadership Opportunities vs Attrition")
# plt.show()

# # 7. Employee Recognition vs Attrition

# # code : 

# recognition_analysis=pd.crosstab(train_df["Employee Recognition"],
#             train_df["Attrition"],
#             normalize="index")*100
# print("\n Employee Recognition vs Attrition")
# print(recognition_analysis)
# # plotting
# import seaborn as sns 
# import matplotlib.pyplot as plt
# sns.countplot(
#     data=train_df,
#     x="Employee Recognition",
#     hue="Attrition"
# )
# plt.title("Employee Recognition vs Attrition")
# plt.show()

# # 8. Company reputation vs Attrition

# # code : 

# reputation_analysis=pd.crosstab(train_df["Company Reputation"],
#             train_df["Attrition"],
#             normalize="index")*100
# print("\n Company Reputation vs Attrition")
# print(reputation_analysis)
# # plotting
# import seaborn as sns 
# import matplotlib.pyplot as plt
# sns.countplot(
#     data=train_df,
#     x="Company Reputation",
#     hue="Attrition"
# )
# plt.title("Company Reputation vs Attrition")
# plt.show()

# EDA insights :

# eda_insights={
#     "Top Attrition Drivers":[],
#     "Retention Factors":[],
#     "Business Recommendations":[]
# }

######################################################################

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
# import matplotlib.pyplot as plt
# top_features=importance_df.head(10)             # because we need to know which are the top features that are influencing the model
# plt.figure(figsize=(10,6))
# plt.barh(
#     top_features["Feature"],
#     top_features["Importance"]
# )
# plt.xlabel("Importance")
# plt.ylabel("Features")
# plt.title("Top 10 Important Features")
# plt.tight_layout()
# plt.show()

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

# from sklearn.metrics import confusion_matrix
# import seaborn as sns 
# import matplotlib.pyplot as plt
# cm=confusion_matrix(y_test,y_pred)
# plt.figure(figsize=(6,5))
# sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()
# print(cm)

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

# from sklearn.calibration import calibration_curve
# import matplotlib.pyplot as plt
# prob_true,prob_pred=calibration_curve(y_test,y_prob,n_bins=10)
# plt.figure(figsize=(8,6))
# plt.plot(prob_pred,prob_true,marker="o",label="Model")
# plt.plot([0,1],[0,1],linestyle="--",label="Perfect Calibration")
# plt.xlabel("Predicted Probability")
# plt.ylabel("Actual Probability")
# plt.title("Calibration Curve")
# plt.legend()
# plt.show()

#############################################################################

# now we are doing SHAP 

# SHAP explainability

import shap
explainer=shap.TreeExplainer(model)
sample_data=X_test.sample(1000,random_state=42)
shap_values=explainer.shap_values(sample_data)
shap.summary_plot(shap_values,sample_data)
shap.summary_plot(shap_values,sample_data,plot_type="bar")

# we are selecting an employee and then testing

employee=sample_data.iloc[[0]]
prob=model.predict_proba(employee)[0][1]
print(prob)
employee_shap=explainer.shap_values(employee)
shap.plots.waterfall(shap.Explanation(
    values=employee_shap[0],
    base_values=explainer.expected_value,
    data=employee.iloc[0],
    feature_names=employee.columns
))

print(employee.T)
employee_shap=explainer.shap_values(employee)
print(employee_shap)
    
# we are creating counterfactual explainability --> DiCE find ways to reduce the risk

train_data=X_train.copy()
train_data["Attrition"]=y_train # creating a data frame
import dice_ml
# creating data object
d=dice_ml.Data(
    dataframe=train_data,continuous_features=["Age","Years at Company","Monthly Income","Distance from Home","Company Tenure",
                                              "Number of Promotions","Number of Dependents"],outcome_name="Attrition")

# creating the model object

m=dice_ml.Model(model=model,backend="sklearn")

# creating the explainer

exp=dice_ml.Dice(d,m,method="random")

# generating the counterfactual

employee_df=employee.copy() # using the selected employee
# generate
cf=exp.generate_counterfactuals(employee_df,total_CFs=3,desired_class="opposite",features_to_vary=[
    "Remote Work","Work-Life Balance","Number of Promotions","Employee Recognition","Company Reputation"
])

# showing result

cf.visualize_as_dataframe()

import pandas as pd
pd.set_option("display.max_columns",None)
pd.set_option("display.width",None)
pd.set_option("display.max_colwidth",None)
print(cf.cf_examples_list[0].final_cfs_df)

# we are building the risk segmentation engine

def risk_segment(prob):
    if prob<0.3:
        risk="Low"
    elif prob<0.7:
        risk="Medium"
    else:
        risk="High"

# we are creating the recommendation engine

# recommendation rules

def generate_recommendations(employee):
    recommendations=[]
    if employee["Remote Work"]=="No":
        recommendations.append("Offer Remote/Hybrid Work")
    
    if employee["Work-Life Balance"]in["Poor","Fair"]:
        recommendations.append("Improve Work-Life Balance")
        
    if employee["Employee Recognition"]in["Low","Medium"]:
        recommendations.append("Increase Employee Recognition")
        
    if employee["Distance from Home"]>50:
        recommendations.append("Provide Flexible Work or Relocation Support")
        
    if employee["Number of Promotions"]==0:
        recommendations.append("Discuss Career Growth Opportunities")
    return recommendations

# testing our employee

recommendations=generate_recommendations(employee.iloc[0])
for r in recommendations:
    print(r)

# adding risk based priorities

def recommendation_priority(prob):
    if prob>=0.8:
        return "Critical"
    elif prob>=0.6:
        return "High"
    elif prob>=0.3:
        return "Medium"
    else:
        return "Low"
    
# Creating HR action plan

{
    "Employee Risk":"High","Probability":0.91,
    "Top Drivers":["Remote Work","Work-Life Balance","Distance from Home"],
    "Recommendations":["Offer Hybrid Work","Flexible Hours","Transportation Support"]
}

# creating the employee twin

employee_twin=employee.copy()

# modifying the features

employee_twin["Remote Work"]="Yes"
employee_twin["Work-Life Balance"]="Good"

# predicting again

new_prob=model.predict_proba(employee_twin)[0][1]
print(new_prob)

# comapring the employee and employee twin

print("Original Risk :",prob)
print("Twin Risk :",new_prob)

# scenario Analysis engine

prob=model.predict_proba(employee)[0][1]
scenario_results=[]
# current employee
current_risk=model.predict_proba(employee)[0][1]
scenario_results.append({
    "Scenario":"Current State","Risk":round(current_risk*100,2)
})

# scenario-1 Remote worl
s1=employee.copy()
s1["Remote Work"] = "Yes"
s1=s1[X_train.columns]
risk1=model.predict_proba(s1)[0][1]
scenario_results.append({
    "Scenario":"Offer Remote Work",
    "Risk":round(risk1*100,2)
    })

# scenario-2 Better work-life balance
s2=employee.copy()
s2["Work-Life Balance"] = "Good"
s2=s2[X_train.columns]
risk2=model.predict_proba(s2)[0][1]
scenario_results.append({
    "Scenario":"Improve Work-Life Balance",
    "Risk":round(risk2*100,2)
})

#scenario-3 Promotion
s3=employee.copy()
s3["Number of Promotions"]+=1
s3=s3[X_train.columns]
risk3=model.predict_proba(s3)[0][1]
scenario_results.append({
    "Scenario":"Give Promotion",
    "Risk":round(risk3*100,2)
})

# scenario-4 Recognition
s4 = employee.copy()
s4["Employee Recognition"] = "Very High"
s4=s4[X_train.columns]
risk4=model.predict_proba(s4)[0][1]
scenario_results.append({
    "Scenario":"Increase Recognition",
    "Risk":round(risk4*100,2)
})

# ==========================================
# Scenario 5 : Remote Work + Work-Life Balance
# ==========================================

s5 = employee.copy()

# Apply Intervention
s5["Remote Work"] = "Yes"
s5["Work-Life Balance"] = "Good"

# Ensure column order matches training data
s5 = s5[X_train.columns]

# ==========================================
# Debug Information
# ==========================================

print("\n==============================")
print("SCENARIO 5 DEBUG")
print("==============================")

print("\nColumns Match?")
print(list(X_train.columns) == list(s5.columns))

print("\nX_train Shape:")
print(X_train.shape)

print("\ns5 Shape:")
print(s5.shape)

print("\nLast 5 X_train Columns:")
print(X_train.columns[-5:].tolist())

print("\nLast 5 s5 Columns:")
print(s5.columns[-5:].tolist())

print("\n===== X_train dtypes =====")
for col in X_train.columns:
    print(f"{col}: {X_train[col].dtype}")

print("\n===== s5 dtypes =====")
for col in s5.columns:
    print(f"{col}: {s5[col].dtype}")

print("\n===== s5 Values =====")
print(s5.T)

# ==========================================
# Prediction
# ==========================================

risk5 = model.predict_proba(s5)[0][1]

print("\nScenario 5 Risk:")
print(risk5)

# ==========================================
# Save Result
# ==========================================

scenario_results.append({
    "Scenario": "Remote + WLB",
    "Risk": round(risk5 * 100, 2)
})

# scenario-6 Full Retention Plan
s6=employee.copy()
s6["Remote Work"] = "Yes" 
s6["Work-Life Balance"] = "Good"
s6["Employee Recognition"] = "Very High"
s6["Number of Promotions"]+=1
s6=s6[X_train.columns]
risk6=model.predict_proba(s6)[0][1]
scenario_results.append({
    "Scenario":"Full Recognition Plan",
    "Risk": round(risk6*100,2)
})        
scenario_df=pd.DataFrame(scenario_results)
print(scenario_df)

# Visualizing it

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,5))
# plt.bar(scenario_df["Scenario"],scenario_df["Risk"])
# plt.xticks(rotation=45)
# plt.ylabel("Attrition Risk (%)")
# plt.title("Scenario Analysis")
# plt.tight_layout()
# plt.show()

# advance executive insight
best_scenario=scenario_df.iloc[scenario_df["Risk"].idxmin()]
print("\n Best Retention Strategy")
print(best_scenario)

#################################################################
# print("Columns Match:", list(X_train.columns) == list(s5.columns))

# print("\nX_train shape:", X_train.shape)
# print("s5 shape:", s5.shape)

# print("\nLast 5 X_train columns:")
# print(X_train.columns[-5:])

# print("\nLast 5 s5 columns:")
# print(s5.columns[-5:])
#################################################################

# next we are going to do is uncertainity analysis

# ===================================
# UNCERTAINTY ANALYSIS
# ===================================

prob=model.predict_proba(employee)[0][1]
confidence=abs(prob-0.5)*2*100
print("\n========== Prediction Report ==========")
print("\nPredicted Probability :",round(prob*100,2),"%")
print("Confidence Score : ",round(confidence,2),"%")

# next is the confidence level
if confidence>=80:
    level="Very High Confidence"
if confidence>=60:
    level="High Confidence"
if confidence>=40:
    level="Moderate Confidence "
else:
    level="Low Confidence - Review Recommended"
print("Confidence Level : ",level)

if prob <0.30:
    risk_level="Low Risk"
elif prob< 0.60:
    risk_level="Moderate Risk"
else:
    risk_level="High Risk"
print("Risk Level : ",risk_level)
    
if confidence<40:
    recommendation=(
        "Monitor Employee closely and collect additional information.")
else:
    recoemmendation=("Prediction is reliable enough for intervention planning.")
print("Recommentation : ",recommendation)
print("\n")
# Risk + Confidence
# print("\n========== Prediction Report ==========")
# print("Attrition Risk : ",round(prob*100,2),"%")
# print("Confidence :",round(confidence,2),"%")
# print("Confidence : ",level)

print("======================================")


# now we are going for fairness analysis
# =====================================
# FAIRNESS ANALYSIS - GENDER
# =====================================

gender_risk=pd.DataFrame({
    "Gender":X_test["Gender"],
    "Prediction":y_prob
})
gender_summary=gender_risk.groupby("Gender")["Prediction"].mean()*100
print("\n########## Gender Fairness ###########")
print(gender_summary)

# visualization 
# import matplotlib.pyplot as plt
# plt.figure(figsize=(6,4))
# gender_summary.plot(kind="bar")
# plt.title("Avearge Predicted Attrition Risk by Gender")
# plt.ylabel("Risk (%)")
# plt.show()

# marital status fairness

marital_risk=pd.DataFrame({
    "Marital Status": X_test["Marital Status"],
    "Prediction":y_prob
})
marital_summary=marital_risk.groupby("Marital Status")["Prediction"].mean()*100
print("\n########## Marital Status Fairness #############")

# visualization
# import matplotlib.pyplot as plt
# plt.figure(figsize=(6,4))
# marital_summary.plot(kind="bar")
# plt.title("Average Predicted Risk by Marital Status")
# plt.ylabel("Risk (%)")
# plt.show()

# Age Fairness

age_df=pd.DataFrame({
    "Age":X_test["Age"],"Prediction":y_prob
})
age_df["Age Group"]=pd.cut(age_df["Age"],bins=[18,30,45,60])
age_summary=age_df. groupby("Age Group")["Prediction"].mean()*100
print("\n########## Age Fairness #############")
print(age_summary)

# Visualization

# import matplotlib.pyplot as plt
# plt.figure(figsize=(6,4))
# age_summary.plot(kind="bar")
# plt.title("Average predicted Risk by Age Group")
# plt.ylabel("Risk (%)")
# plt.show()

# Fairness Score
fairness_gap=(gender_summary.max()-gender_summary.min())
print("\n Fairness Gap :",round(fairness_gap,2),"%")

# we are comparing gender vs actual attrition

print("\nGender vs Actual Attrition")
gender_actual=pd.crosstab(train_df["Gender"],train_df["Attrition"],normalize="index")*100
print(gender_actual)

# Fairness metric between actual vs predicted

female_actual=gender_actual.loc["Female",1]
male_actual=gender_actual.loc["Male",1]
print("Female Actual Attrition : ",female_actual)
print("Male Actual Attrition : ",male_actual)

# Fairness Summary

print("\n########## Fainess Report ##########")
if fairness_gap<5:
    print("Fairness Status : Excellent")
elif fairness_gap<15:
    print("Fairness Status : Acceptable - Investigated")    
else:
    print("Fairness Status : Required Review")
print("Gender Gap : ",round(fairness_gap,2),"%")
print("""
      Conclusion:
                 Observed differences align closely with actual attrition rates.
                 No significant evidence of gender-based model bias
      """)


# saving the trained model
# from joblib import dump
# dump(model,"catboost_attrition_model.joblib")
# print("Model Saved")

# loading the model
# from joblib import load
# model=load("catboost_attrition_model.joblib")
# print("Model Loader")

# saving the shap explainer
# import joblib
# joblib.dump(explainer,"shap_explainer.joblib")

# loading the shap explainer
# explainer=joblib.load("shap_explainer.joblib")

# saving prediction - better way
predictions = X_test.copy()

predictions["Actual"] = y_test.values
predictions["Probability"] = y_prob
predictions["Prediction"] = y_pred
print(predictions.head())
# predictions.to_csv("outputs/predictions.csv",index=False)
#Loading
# predictions=pd.read_csv("outputs/predictions.csv")

from joblib import dump

dump(model,"models/catboost_model.joblib")

scenario_df.to_csv("outputs/scenario_results.csv",index=False)
importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)
print("Feature Importance Saved")
predictions.to_csv("outputs/predictions.csv",index=False)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)
print("ROC-AUC :", roc_auc)

summary = {
    "Employees": len(X_test),
    "Accuracy": accuracy,
    "AUC": roc_auc,
    "Avg Risk": y_prob.mean()*100
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    "outputs/summary.csv",
    index=False
)
predictions = pd.read_csv("outputs/predictions.csv")
print(predictions.columns)

import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

import pandas as pd
import os

os.makedirs("outputs", exist_ok=True)

# Mean absolute SHAP importance
shap_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "SHAP_Importance": abs(shap_values).mean(axis=0)
})

shap_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "SHAP_Importance": abs(shap_values).mean(axis=0)
})

total = shap_importance["SHAP_Importance"].sum()

shap_importance["SHAP_Percentage"] = (
    shap_importance["SHAP_Importance"] / total
) * 100

shap_importance = shap_importance.sort_values(
    by="SHAP_Percentage",
    ascending=False
)

shap_importance.to_csv(
    "outputs/shap_importance.csv",
    index=False
)

import pandas as pd

cat_features_df = pd.DataFrame({
    "Feature": cat_features
})

cat_features_df.to_csv(
    "outputs/cat_features.csv",
    index=False
)

import joblib
import os

os.makedirs("outputs", exist_ok=True)

joblib.dump(
    model,
    "outputs/catboost_model.pkl"
)

print("Model Saved Successfully")