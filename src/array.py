import numpy as np
import scipy.io

def rflatten_obj(A):

    if A.dtype == 'O':
        dim = np.shape(A)
        n = len(dim)
        ad = np.zeros(n)
        i = 0
        tmp = []
        for a in A:
            tmp.append(rflatten_obj(a))
        return_val = np.concatenate(tmp)

    else:
        return_val = A.flatten()

    return return_val

c1 = scipy.io.loadmat('data-600.mat')['c1'][0]
c2 = rflatten_obj(c1)

print np.shape(c2)
print c2[2723]
print c2[4323]
print c2[4723]
print c2[473]
print c2[4724]

'''
while True:
    if ad[i] == dim[i]:
        if i == n - 1:
            break;
        ad[i] = 0
        i += 1
    ad[i] += 1
'''
