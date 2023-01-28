import numpy as np
import matplotlib.pyplot as plt
import pywt

s = np.linspace(-4, 4, 1000)

s_soft = pywt.threshold(s, value=1, mode='soft')
s_hard = pywt.threshold(s, value=1, mode='hard')

# plot results, label the axis and add a legend
plt.plot(s, s_soft, label='soft')
plt.plot(s, s_hard, label='hard')
plt.legend()

#label axis
plt.xlabel('input value')
plt.ylabel('thresholded value')
# plt.show()

# Save plot to pdf
plt.savefig('img/soft-hard-threshold.pdf')
