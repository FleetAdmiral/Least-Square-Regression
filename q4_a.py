# Pranav Nair 201525149
from numpy import genfromtxt
import sys

#initializing the lasso model
from sklearn import linear_model

def train_lasso(train):
    reg = linear_model.Lasso(alpha = 0)

    #getting the length of each input
    train_len_row = len(train[0])

    #getting number of rows in input
    train_len = len(train)

    i=0
    ans = []
    #train_real = [][train_len_row]
    w, h = train_len_row-1, train_len
    train_real = [[0 for x in range(w)] for y in range(h)]

    while (i < train_len):  # to get the first column:- 1 or 0 value
        j=0
        while(j<train_len_row-1):
            train_real[i][j]=train[i][j]
            j=j+1
        ans.append(train[i][train_len_row-1])
        i = i + 1

    reg.fit(train_real, ans)
    return reg

def predict(reg, test):
    # uncomment the below if the last column is the answers
    test_len = len(test)
    test_len_row = len(test[0])
    test_real = [[0 for x in range(test_len_row - 1)] for y in range(test_len)]
    i = 0
    ans=[]
    while (i < test_len):
        j = 0
        ans.append(test[i][test_len_row-1])
        while (j < test_len_row - 1):
            test_real[i][j] = test[i][j]
            j = j + 1
        i = i + 1

    prediction= reg.predict(test_real)
    i=0
    prediction_len = len(prediction)
    count=0
    while(i<prediction_len):
        if(prediction[i]>=0.5):
            print(1)
            if(ans[i]==1):
                count=count+1
        else:
            print(0)
            if (ans[i] == 0):
                count = count + 1
        i=i+1


#importing test and train files
train = genfromtxt(sys.argv[1], delimiter=',')
test = genfromtxt(sys.argv[2], delimiter=',')

reg_lasso = train_lasso(train)

#comment the below if the last column is answers

predict(reg_lasso, test)