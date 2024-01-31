import pandas as pd
import numpy as np

# Take data.csv and split into train, validation, test sets
def splitData():
    data = pd.read_csv('data.csv', header=0)
    data = data.to_numpy()
    print(data)
    np.random.shuffle(data)
    train, validate, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])
    print(train)
    np.savetxt('train.csv', train, delimiter=',')
    np.savetxt('validate.csv', validate, delimiter=',')
    np.savetxt('test.csv', test, delimiter=',')

if __name__ == '__main__':
    print("Splitting data...")
    splitData()
