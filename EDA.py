import pandas as pd
import numpy as np

# we are going to load the dataset

train_df=pd.read_csv("Dataset/train.csv")
test_df=pd.read_csv("Dataset/test.csv")

# printing the shape of each csv files

print("Train csv shape",train_df.shape) # we are doing this to know how many rows and columns are there
print("Test csv shape",test_df.shape)

# next we are going to check how the data looks like head() -> prints first 5 rows, we will get an idea about how the data is

print(train_df.head())  # there are 24 features present

# next we need to check column informations

print(train_df.info())  # to check column names, data types and missing values

# next we are printing statistical summary --> describe()

print(train_df.describe()) # this will give min,max,mean,median,std

# Missing value analysis

missing_values=train_df.isnull().sum()
print(missing_values)                         # well it looks like there are no missing values.

# now we are checking duplicate values

duplicates=train_df.duplicated().sum()
print("Duplicate values",duplicates)         # and there are no duplicate values as well.

# target column discovery

print(train_df.columns)  # there are 24 columns

# Target distribution

target="Attrition"
print(train_df[target].value_counts())    # this is to understand class balance we have 31k --> stayed and 28k --> left

# Target percentage

print(train_df[target].value_counts(normalize=True)*100) # we are checking because accuracy can be misleading sometimes

# we are seperating numerical and categorical values

numerical_cols=train_df.select_dtypes(include=np.number).columns.tolist()
categorical_cols=train_df.select_dtypes(include="object").columns.tolist()
print("Numerical Features : ")                                 
print(numerical_cols)               # because later we will be using it 
print("\nCategorical Features : ")               # what we have done here is seperated categorical and numerical columns
print(categorical_cols)

# we are saving the EDA report 

eda_report={
    "Train shape": train_df.shape,
    "Test shape": test_df.shape,
    "Miising Values": train_df.isnull().sum().sum(),
    "Duplicates": train_df.duplicated().sum(),
    "Numerical Columns": len(numerical_cols),
    "Categorical Columns": len(categorical_cols)
}
print(eda_report)

# droping the employe id because we dont need that for training

train_df.drop("Employee ID",axis=1)

##########################################################################################
#################### Now we are going to answer some EDA questions########################

1. Does age affect Attrition ? 

#code : 

import seaborn as sns 
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
sns.histplot(data=train_df,x="Age",hue="Attrition",kde=True,multiple="stack")
plt.title("Age vs Attrition")
plt.show()

# 2. Monthly income vs Attririon 

# code :

plt.figure(figsize=(10,6))
sns.boxplot(data=train_df,x="Attrition",y="Monthly Income")
plt.title("Monthly income vs Attrition")
plt.show()

# 3. Over time vs Attrition

# code :

overtime_analysis=pd.crosstab(train_df["Overtime"],
            train_df["Attrition"],
            normalize="index")*100
print("\n Over Time vs Attrition")
print(overtime_analysis)
# plotting
import seaborn as sns 
import matplotlib.pyplot as plt 
plt.figure(figsize=(8,5))
sns.countplot(data=train_df,x="Overtime",hue="Attrition")
plt.title("Overtime vs Attrition")
plt.show()

# 4. Job satisfaction vs Attrition

# code :

job_satisfaction_analysis=pd.crosstab(train_df["Job Satisfaction"],
            train_df["Attrition"],
            normalize="index")*100
print("\n Job satisfaction vs Attrition")
print(job_satisfaction_analysis)
# plotting
import seaborn as sns 
import matplotlib.pyplot as plt 
plt.figure(figsize=(10,5))
sns.countplot(data=train_df,
              x="Job Satisfaction",
              hue="Attrition")
plt.title("Job Satisfaction vs Attrition")
plt.show()  

# 5. Work-life balance vs Attrition

# code :

wlb_analysis=pd.crosstab(train_df["Work-Life Balance"],
            train_df["Attrition"],
            normalize="index"
)*100
print("\n Work-Life Balance Analysis vs Attrition ")
print(wlb_analysis)
# plotting
import seaborn as sns 
import matplotlib.pyplot as plt
sns.countplot(
    data=train_df,
    x="Work-Life Balance",
    hue="Attrition"
)
plt.title("Work-Life Balance vs Attrition")
plt.show()

# 6. Leadership Oppertunites vs Attrition

# code :

leadership_analysis=pd.crosstab(train_df["Leadership Opportunities"],
            train_df["Attrition"],
            normalize="index")*100
print("\n Leadership Oppertunities vs Attrition")
print(leadership_analysis)
 # plotting
import seaborn as sns 
import matplotlib.pyplot as plt
sns.countplot(
    data=train_df,
    x="Leadership Opportunities",
    hue="Attrition"
) 
plt.title("Leadership Opportunities vs Attrition")
plt.show()

# 7. Employee Recognition vs Attrition

# code : 

recognition_analysis=pd.crosstab(train_df["Employee Recognition"],
            train_df["Attrition"],
            normalize="index")*100
print("\n Employee Recognition vs Attrition")
print(recognition_analysis)
# plotting
import seaborn as sns 
import matplotlib.pyplot as plt
sns.countplot(
    data=train_df,
    x="Employee Recognition",
    hue="Attrition"
)
plt.title("Employee Recognition vs Attrition")
plt.show()

# 8. Company reputation vs Attrition

# code : 

reputation_analysis=pd.crosstab(train_df["Company Reputation"],
            train_df["Attrition"],
            normalize="index")*100
print("\n Company Reputation vs Attrition")
print(reputation_analysis)
# plotting
import seaborn as sns 
import matplotlib.pyplot as plt
sns.countplot(
    data=train_df,
    x="Company Reputation",
    hue="Attrition"
)
plt.title("Company Reputation vs Attrition")
plt.show()

EDA insights :

eda_insights={
    "Top Attrition Drivers":[],
    "Retention Factors":[],
    "Business Recommendations":[]
}