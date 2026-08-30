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

  ![Model Results](\salary vs education level.png)
- Since the influence is linear between Salary and Education Level we can rank the Education Level with numerics

```python
  education_rank = {"Bachelor's":0,"Master's":1,"PhD":2}
  data['education'] = data['Education Level'].map(education_rank)
```

  This code will map the previous categorical data in education column with numerical data

- Performed OneHotEncoding on Gender column and dropped the first column to avoid hallucination
- Refined the job role into grouped one's because the previous job role has more number of job roles
- Reduced the number of 174 job roles to 40 job roles with one others category
- Performed preprocessing using OneHotEncoding to convert reduced job roles into numerical data
- Split the entire dataset into X and Y and checked the shape of the individual
- Split the individual X and Y into train and test splits
- Later fit the train data into the model and predicted the salary
- Analysed the scores based on the prediction data

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run
1. Clone the repo
