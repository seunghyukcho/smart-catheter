import numpy as np
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVR, SVR


print("** loading train data.. **")
scales = np.load("./data/proc/V2/train/scales.npy")
signals = np.load("./data/proc/V2/train/signals.npy")
print(np.shape(signals), np.shape(scales))

scales, signals = scales.reshape((-1,1)), signals.reshape((-1, 300))
signals = (signals - signals[0])
print(np.shape(signals), np.shape(scales))
print()


print("** building model.. **")
mode = 'poly'
gamma = 'auto'
degree = 7
coeff = 1.0
epsilon = 1.0
nonlin_svm = SVR(kernel=mode, gamma=gamma, degree=degree, C=coeff, epsilon=epsilon)
print("[model parameters]")
print(f"mode: {mode}")
print(f"gamma: {gamma}")
print(f"degree: {degree}")
print(f"coeff: {coeff}")
print(f"epsilon: {epsilon}")
print()


print("** training model.. **")
print()
nonlin_svm.fit(signals, scales.ravel())


print("** loading test data.. **")
scales = np.load("./data/proc/V2/test/scales.npy")
signals = np.load("./data/proc/V2/test/signals.npy")
scales, signals = scales.reshape((-1,1)), signals.reshape((-1, 300))
signals = (signals - signals[0])
print("[label statistics]")
print(f"len: {len(scales)}")
print(f"mean: {np.mean(scales)}")
print(f"std: {np.std(scales)}")
print()


print("** evaluating.. **")
pred = nonlin_svm.predict(signals)
print("[prediction statistics]")
print(f"len: {len(pred)}")
print(f"mean: {np.mean(pred)}")
print(f"std: {np.std(pred)}")
print(f"l1 loss: {np.mean( np.abs( pred - scales ) )} ")
print(f"error mean: {np.mean( pred - scales  )} ")
print(f"error std: {np.std( pred - scales )} ")
print()


print("** plotting figure.. **")
plt.figure(figsize=(9,4))
plt.plot(scales, 'r')
plt.plot(pred, 'b')
plt.show()