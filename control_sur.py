import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PINN.tf_controller import TFController, USat
from heating_rod_model import HRModel
from time import time


# Import Data
with open('data.pkl', 'rb') as file:
    XR, Temp_lb, Temp_ub, Exact, tm, n_l = pickle.load(file)
file.close()

# Import System and model
hr_model = HRModel()
model = tf.keras.models.load_model('model.h5')


m = 1
cont = TFController(model=model, m=m)
cont.model.u = tf.Variable(m*[1e-6], trainable=True, constraint=USat(0, 1.25))
cont.model.u_old = cont.model.u.numpy()

x0 = [3/30]
x, u, xp = [x0[0]], [1e-6], [x0[0]]

sps = [0.5, 0.4, 0.8, 0.55, 0.3, 0.5]
ds = 25
sp = np.ones(len(sps)*ds)
i = 0
for (i, sp_i) in enumerate(sps):
    sp[i*ds:ds*(i+1)] = sp_i
    i += 1

t0 = time()
for k in range(len(sp)):
    cont.iter = 0
    cont.solve_with_tf_optimizer(x=x0, sp=sp[k], n_step=5000)

    x0_new = hr_model.run(Q=cont.model.u.numpy()[0]*1.25)
    xp.append(cont.model(tf.stack([[cont.model.u[0]], x0, [1.0]], axis=1)).numpy()[0][0])
    x0 = [(x0_new-Temp_lb)/(Temp_ub-Temp_lb)]
    x.append(x0[0])
    u.append(cont.model.u.numpy()[0])
    cont.model.u_old = cont.model.u.numpy()

t1 = time()

print('Req. Time: ', t1-t0)
# Visualization
X = np.array(x)*(Temp_ub-Temp_lb) + Temp_lb
Xp = np.array(xp)*(Temp_ub-Temp_lb) + Temp_lb
SP = np.array(sp)*(Temp_ub-Temp_lb) + Temp_lb
t = list(range(len(u)))

plt.subplot(2, 1, 1)
plt.step(t[:-1], SP, 'r-.', label='Sp')
plt.plot(t, X, label='CV')
plt.plot(t, Xp, 'y:', label='CV Pred')
plt.ylabel('Temperature (C)')
plt.legend()

plt.subplot(2, 1, 2)
plt.step(t, u, 'g')
plt.xlabel('Time')
plt.ylabel('Heater Duty')
plt.show()
