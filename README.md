# BigDataAnalysis
# Problem Statement:
A big data analysis tool is very rudimentary to any big data analysis tasks. In this project, I have create and developed the big data analysis script that is able to perform data analysis automatically by leveraging a machine learning techqniue. Specifically, this tool enables us to train a machine-learning model unsupervisedly and applied it to cluster unseen data samples. 

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

2. K-means:
K-means is a classic unsupvised learning algorithm that enables us to automatically cluster all the data points without having any class labels. By doing this, the algorithm can group all the data points together based on similar patterns in the fatures. The K-mean algorithms is the following: <br />
(1) Randomly choose K samples from the training data as the initial centroids. <br />
(2) For every data point, calculate an euclidean distance between the data point and the centroids. The algorithm then assigns every point to the associated centroid based on the minimum distance. <br />
(3) Update all the centroids. <br />

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
I have tested this tool multiple times, and it is stable to process over 350,000 data examples. For prediction, the algorithm is also able to make a prediction a large number of testing data. During training, there are several important observations. First, a large number of centroids increases the epoch number to converge. In other words, the algorithm needs more epochs to find optimal centroids. Second, a large amount of training data also causes the algorithm needs more epochs to converge. When the number of data points is huge, the algorithm will need more epochs to adjust its centroids to fit extra training examples.
