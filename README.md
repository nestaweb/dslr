# DSLR

1. Prog Describe.py
- Show data for all courses for each hogwarts house
	- Count
	- mean
	- std
	- min
	- 25%
	- 50%
	- 75%
	- max

2.1. Histogram for each course
- y : Nb student
- x : Mark 
- Color : Hogwarts House

- Question : Which Hogwarts course has a homogeneous score distribution between all four houses?

2.2 Scatter plot for each course
- Check the correlation between the marks of the different courses
- Question : What are the two features that are similar?
- x : One course
- y : One other course
- Color : Hogwarts House

2.3. Pair plot
- Question : From this visualization, which features are you going to use for your logistic regression?
- X : All courses
- y : All courses
- When same course : Show Histogram
- When different course : Show Scatter plot


3. Logistic Regression
- logreg_train
	- use dataset_train.csv
	- gradient descent
	- generate weights

- logreg_predict
	- use dataset_test.csv
	- generate houses.csv