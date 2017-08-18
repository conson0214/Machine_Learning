from svmutil import *
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np

def plot_roc(deci, label, title):
    # count of postive and negative labels
    db = []
    pos, neg = 0, 0
    for i in range(len(label)):
        if label[i] > 0:
            pos += 1
        else:
            neg += 1
        db.append([deci[i], label[i]])

    # sorting by decision value
    db = sorted(db, key=itemgetter(0), reverse=True)

    # calculate ROC
    xy_arr = []
    tp, fp = 0., 0.  # assure float division
    for i in range(len(db)):
        if db[i][1] > 0:  # positive
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    # area under curve
    aoc = 0.
    prev_x = 0
    for x, y in xy_arr:
        if x != prev_x:
            aoc += (x - prev_x) * y
            prev_x = x

    xy_unzip = zip(*xy_arr)
    plt.figure()
    plt.plot(np.array(xy_unzip[0]), np.array(xy_unzip[1]), label="", color="red", linewidth=2)
    # print np.array(xy_unzip[0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve of %s (AUC = %.4f)" % (title, aoc))
    plt.savefig('ROCCurve.png')
    plt.show()

def main():
    test_file = r'test.txt.scale'
    model_file = r'train.txt.model'
    test_y, test_x = svm_read_problem(test_file)
    model = svm_load_model(model_file)
    py, evals, deci = svm_predict(test_y, test_x, model)
    labels = model.get_labels()
    deci = [labels[0] * val[0] for val in deci]
    plot_roc(deci, test_y, 'AE')

if __name__ == '__main__':
    main()