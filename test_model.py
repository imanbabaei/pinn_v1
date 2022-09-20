import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('data.pkl', 'rb') as file:
    XR, Temp_lb, Temp_ub, Exact, tm, n_l = pickle.load(file)
file.close()

model = tf.keras.models.load_model('model.h5')

yp = model(XR)
# yp = yp.numpy().reshape((n_t, n_x))

# plt.plot(yp[:, 84])
yp = yp*(Temp_ub-Temp_lb)+Temp_lb
yp = np.reshape(yp, (200, 100))

plt.figure()
# plt.plot(t, Th.value, 'r-', label=r'$T_{heater}\,(^oC)$')
# plt.plot(tm, Exact, 'r', label=r'$T_5\,(^oC)$')
t_train = tm[:n_l+1]
t_valid = tm[n_l:]

plt.plot(t_train, Exact[:n_l+1, 5], 'g', label=r'$T_5\,(^oC)$')
plt.plot(t_train, Exact[:n_l+1, 25], 'g', label=r'$T_{15}\,(^oC)$')
plt.plot(t_train, Exact[:n_l+1, 50], 'g', label=r'$T_{25}\,(^oC)$')
plt.plot(t_train, Exact[:n_l+1, 75], 'g', label=r'$T_{45}\,(^oC)$')
plt.plot(t_train, Exact[:n_l+1, -1], 'g', label=r'$T_{tip}\,(^oC)$')

plt.plot(t_valid, Exact[n_l:, 5], 'r', label=r'$T_5\,(^oC)$')
plt.plot(t_valid, Exact[n_l:, 25], 'r', label=r'$T_{15}\,(^oC)$')
plt.plot(t_valid, Exact[n_l:, 50], 'r', label=r'$T_{25}\,(^oC)$')
plt.plot(t_valid, Exact[n_l:, 75], 'r', label=r'$T_{45}\,(^oC)$')
plt.plot(t_valid, Exact[n_l:, -1], 'r', label=r'$T_{tip}\,(^oC)$')

# plt.plot(tm, yp, 'k--', label=r'$T_5\,(^oC)$')

plt.plot(tm, yp[:, 5], 'k--', label=r'$T_5\,(^oC)$')
plt.plot(tm, yp[:, 25], 'k:', label=r'$T_{15}\,(^oC)$')
plt.plot(tm, yp[:, 50], 'k:', label=r'$T_{25}\,(^oC)$')
plt.plot(tm, yp[:, 75], 'k-.', label=r'$T_{45}\,(^oC)$')
plt.plot(tm, yp[:, -1], 'b-', label=r'$T_{tip}\,(^oC)$')
plt.ylabel(r'$T\,(^oC$)')
plt.xlabel('Time (min)')
# plt.xlim([0, 50])
plt.legend(loc=4)
plt.show()
