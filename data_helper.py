import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_simulation_data(length, type, num_sample):
    samples = np.random.randn(num_sample, length)
    coeffs = np.random.random([length])
    bias = np.random.random()
    labels = np.zeros(num_sample)
    for i in range(num_sample):
        for j in range(length):
            labels[i] += (np.power(samples[i, j], coeffs[j]))
        labels[i] += bias
        if type == 'classification':
            labels[i] = 1.0 if sigmoid(labels[i]) > 0.5 else 0.0
        else:
            labels[i] = labels[i] * 10.0

    return samples, labels, coeffs, bias


def print_feature_importance(pred, act):
    pred_sorted = np.sort(np.copy(pred))
    pred_sorted_index = dict(zip(pred_sorted, np.arange(np.shape(pred_sorted)[0])))
    act_sorted = np.sort(np.copy(act))
    act_sorted_index = dict(zip(act_sorted, np.arange(np.shape(act_sorted)[0])))
    compare = list(zip(pred, act))
    compare_all = [(i[0], i[1], pred_sorted_index[i[0]], act_sorted_index[i[1]]) for i in compare]

    return compare_all



if __name__ == '__main__':
    print('hello')
    a = make_simulation_data(3, 'classification', 20)
    # print(a)
    # c = np.random.random([3,4])
    # print(c[0])
    # b = [i for i in c]
    # print(b)

    print_feature_importance([1,4,3,2,5], [1,3,2,4,5])