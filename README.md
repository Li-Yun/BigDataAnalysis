# BigDataAnalysis
# Problem Statement:

# Techniques:
In order to accomplish the data clustering tas, I pre-processed the input data by converting symbolic features to numerical features in different ways and utilized a classic unsuprivsed learning algorithm, k-means, to automatically analyze the unseen data.

1. Data pre-processing: <br />
Since most of the features are symbolic representations, it is easy to conver all of them to numerical features for further usage. I am going to show how I cast different features to the numerical representations. <br />
(1) **time_created**: I discard this feature since it is similar to feature "date_created". <br />
(2) **date_created**: I convert the date information to weeks of the difference betwee today and a specific date. <br />
(3) **up_votes**: I directly cast strings to integers for this feature. <br />
(4) **down_votes**: I also directly cast strings to integers for this feature. <br />
(5) **title**: I convert a sentence to the bag-of-words feature and calculate an average distinct word for a given bag-of-words feature. <br />
(6) **author**: I represent the name of authors as length of characthers. For example, name "polar" becomes 5 <br />
(7) **category**: I create a dictionary structure to map a string to the associated number. <br />

2. K-means: <br />
==> afternoon



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
