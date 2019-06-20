# BigDataAnalysis
# Problem Statement:

# Techniques:


# How to run the script?
1. Please download a dataset for script running. <br />
Link: https://drive.google.com/file/d/15X00ZWBjla7qGOIW33j8865QdF89IyAk/view?usp=sharing\

2. Training the k-means model:
```
python3 main.py --train True [--train_ratio 0.7][--K 7][--max_itr 2000][--tol_num 1e-5]

For example:
python3 main.py --train True
```

3. Predicting the new data and automatically perform data clustering
```
python3 main.py --predict True [--train_ratio 0.7][--K 7][--max_itr 2000][--tol_num 1e-5]

For example:
python3 main.py --predict True
```

# Conclusion:
