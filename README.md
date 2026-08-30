# Salary Prediction using Linear Regression

## Overview
This is the very first project of mine in the concept of machine learning, in sub-concept Linear Regression, here I have developed a linear regression model which predicts the salary of an individual.

## Dataset
- Source: salary.csv (from kaggle)
- Description
- Rows: Age, Gender, Education Level, Job Title, Years of Experience and salary

## Project Workflow
- Beginning the work with reading the dataset
- Analysed the statical numbers in the data for Age, Years of Experience and Salary
- Found 2 rows that have null values, since the null rows are just 2 I have dropped these columns which don't affect the data
- Since there are categorical values in Educational Levels we must convert them into numeric data
- Before converting the categorical data we must ensure that there is a positive influence on Salary by Education Level

  ![Model Results](salary%20vs%20education%20level.png)
- Since the influence is linear between Salary and Education Level we can rank the Education Level with numerics

```python
  education_rank = {"Bachelor's":0,"Master's":1,"PhD":2}
  data['education'] = data['Education Level'].map(education_rank)
```

  This code will map the previous categorical data in education column with numerical data

- Performed OneHotEncoding on Gender column and dropped the first column to avoid hallucination
```python
  pd.get_dummies(data,columns=['Gender'],drop_first=True)
```
- Refined the job role into grouped one's because the previous job role has more number of job roles
- Reduced the number of 174 job roles to 40 job roles with one others category
```python
  title_counts = data['Job Title'].value_counts()
  rare_titles = title_counts[title_counts<3].index

  data['job title grouped'] = data['Job Title'].apply(lambda x : 'rare' if x in rare_titles else x )
```
- Performed preprocessing using OneHotEncoding to convert reduced job roles into numerical data
```python
  from sklearn.preprocessing import OneHotEncoder

  encoder = OneHotEncoder(sparse_output=False, drop='first',handle_unknown='ignore')

  encoded_array = encoder.fit_transform(data[['job title grouped']])

  encoded_cols = encoder.get_feature_names_out(['job title grouped'])

  jobs = pd.DataFrame(encoded_array, columns=encoded_cols)

  final = pd.concat([data,jobs],axis=1)
```
- Split the entire dataset into X and Y and checked the shape of the individual
- Split the individual X and Y into train and test splits
- Later fit the train data into the model and predicted the salary
- Analysed the scores based on the prediction data
```python
  from sklearn.metrics import mean_absolute_error,mean_squared_error
  mae = mean_absolute_error(y_test,y_pred)
  print(f'Mean Absolute Error : {mae}')
  mse = mean_squared_error(y_test,y_pred)
  print(f'Mean Squared Error : {mse}')
  print(f'R^2 Score : {score}')
  Output :
  Mean Absolute Error : 12699.34433502276
  Mean Squared Error : 314650189.85482186
  R^2 Score : 0.8581993242086992
```
- Prediction analysis based on salary feature
  ```python
  mlp.figure(figsize=(10,8))
  mlp.scatter(x_test_sorted[:,0],y_test,label='Actual Data',linestyle='--')
  mlp.plot(x_test_sorted[:,0],y_pred_sorted,label='Predicted Data',c='r')
  mlp.legend()
  mlp.xlabel('Age')
  mlp.ylabel('Salary')
  ```
  ![Model Result](prediction%20analysis.png)

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run
1. Clone the repo

# Author

Puli Abhinav

B.Tech – Computer Science & Engineering (Data Science)

Vaagdevi Engineering College

puliabhinav3@gmail.com
