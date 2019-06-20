# BigDataAnalysis
# Problem Statement:

# Techniques:
In order to accomplish the data clustering tas, I pre-processed the input data by converting symbolic features to numerical features in different ways and utilized a classic unsuprivsed learning algorithm, k-means, to automatically analyze the unseen data.

1. Data pre-processing: <br />
Since most of the features are symbolic representations, it is easy to conver all of them to numerical features for further usage. I am going to show how I cast different features to the numerical representations. <br />
(1) time_created: 
(2) date_created:


2. K-means: <br />


time_created,date_created,up_votes,down_votes,title,over_18,author,category






# How to run the script?
1. Please download a dataset for script running. <br />
Link: https://drive.google.com/file/d/15X00ZWBjla7qGOIW33j8865QdF89IyAk/view?usp=sharing\

2. Training the k-means model:
```
python3 main.py --train True [--train_ratio 0.7][--K 7][--max_itr 2000][--tol_num 1e-5]

For example:
python3 main.py --train True
```

3. Predicting the new data and automatically performing data clustering
```
python3 main.py --predict True [--train_ratio 0.7][--K 7][--max_itr 2000][--tol_num 1e-5]

For example:
python3 main.py --predict True
```

# Conclusion:
