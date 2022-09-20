from pickle import dump, load
from PINN.pinn_solver import PINNSolver
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

DTYPE = 'float64'
tf.random.set_seed(12345)
np.random.seed(12345)


class HRPINN(PINNSolver):
    def __init__(self, x_r, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store Model Variables
        # self.model.alpha = tf.Variable(initial_value=8e-6,
        #                                trainable=True,
        #                                dtype=DTYPE,
        #                                constraint=tf.keras.constraints.NonNeg())
        #
        # self.model.beta = tf.Variable(initial_value=2e-3,
        #                               trainable=True,
        #                               dtype=DTYPE,
        #                               constraint=tf.keras.constraints.NonNeg())

        # Store Model Constants
        self.alpha = None
        self.beta = None
        # self.Ts = None

        # Store collocation points
        self.qh = x_r[:, 0:1]
        self.theta = x_r[:, 1:2]
        self.x = x_r[:, 2:3]

        # self.x_u = self.x*(x_ub-x_lb) + x_lb
        # self.t_u = self.t*(t_ub-t_lb)+t_lb

    def get_residual_loss(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t and x during this GradientTape
            tape.watch(self.x)

            # Compute current values y(t,x)
            y = self.model(tf.stack([self.qh[:, 0], self.theta[:, 0], self.x[:, 0]], axis=1))
            y_x = tape.gradient(y, self.x)

        # y_t = tape.gradient(y, self.t)
        y_xx = tape.gradient(y_x, self.x)

        del tape

        return self.residual_function(y, y_x, y_xx)

    def residual_function(self, y, y_x, y_xx):
        """Residual of the PDE"""
        y = y*30+20
        y_x = y_x*30
        y_xx = y_xx * 30
        # y_x = (y[:, 1:]-y[:, :-1])/0.001
        # y_xx = (y_x[:, 1:]-y_x[:, :-1])/0.001
        y_t = (y[1:, :]-y[:-1, :])/3015.07537
        # tf.print(y)
        # res = y_t[2:]*30 + self.model.beta * (y[2:, 1:]*30+20-23) - self.model.alpha * y_xx[:, 1:]*30
        res = y_t - self.alpha * y_xx[:-1] + self.beta * (y[:-1]-23)

        # res[0] -= qh
        # return y_t/3000.0*100.0 - self.model.alpha * y_xx/(0.1-0.001)**2*100 + self.model.beta * (y*100+20 - self.Ts)
        return res

    def callback(self, *args):
        if self.iter % 10 == 0:
            # print('It {:05d}: loss = {:10.8e}, alpha = {:10.4e}, beta = {:10.4e}'.format(self.iter,
            print('It {:05d}: loss = {:10.8e}'.format(self.iter,
                                                      self.current_loss,)
                                                      # self.model.alpha.numpy(),
                                                      # self.model.beta.numpy())
                  )
        self.hist.append(self.current_loss)
        self.iter += 1


# Load Data
with open('../heating rod/full_data_Q.pkl', 'rb') as file:
    t, x, Qh, Temp, rho, cp, heff, keff, Ts, A, As, nt, L_seg, L, seg, T_melt, pi, d = load(file)
    file.close()
Temp = np.array(Temp)
Qh = np.array(Qh)

n_t, n_x = len(t), len(x)

N_u = 140

layers = [3, 3, 1]

# Domain bounds
t_lb, t_ub = t.min(0), t.max(0)
x_lb, x_ub = x.min(0), x.max(0)
Temp_lb, Temp_ub = (20, 50)
Qh_lb, Qh_ub = (0, 1.25)

# Normalize Data
x = (x - x_lb) / (x_ub - x_lb)
t = (t - t_lb) / (t_ub - t_lb)
Temp = (Temp - Temp_lb) / (Temp_ub - Temp_lb)
Qh = (Qh - Qh_lb) / (Qh_ub - Qh_lb)

t = t.flatten()[:, None]
x = x.flatten()[:, None]
Qh = Qh.flatten()[:, None]

Exact = np.real(Temp).T

Theta = Temp[-1, :].copy()
Theta[1:] = Theta[0: -1]
# Qh[:-1] = Qh[1:]

n_l = 70

X, T = np.meshgrid(x, t)
_, QH1 = np.meshgrid(x, Qh[:n_l])
_, THETA1 = np.meshgrid(x, Theta[:n_l])

_, QH2 = np.meshgrid(x, Qh[n_l:])
_, THETA2 = np.meshgrid(x, Theta[n_l:])

_, QH = np.meshgrid(x, Qh)
_, THETA = np.meshgrid(x, Theta)

x_train = np.hstack((QH1.flatten()[:, None], THETA1.flatten()[:, None]))
x_train = np.hstack((x_train, X[:n_l].flatten()[:, None]))
y_train = Exact[:n_l].flatten()[:, None]

x_valid = np.hstack((QH2.flatten()[:, None], THETA2.flatten()[:, None]))
x_valid = np.hstack((x_valid, X[n_l:].flatten()[:, None]))
y_valid = Exact[n_l:].flatten()[:, None]

XR = np.hstack((QH.flatten()[:, None], THETA.flatten()[:, None]))
XR = np.hstack((XR, X.flatten()[:, None]))


# idx = np.random.choice(N_u, N_u, replace=False)

x_train = tf.convert_to_tensor(x_train, dtype=DTYPE)
y_train = tf.convert_to_tensor(y_train, dtype=DTYPE)

x_valid = tf.convert_to_tensor(x_valid, dtype=DTYPE)
y_valid = tf.convert_to_tensor(y_valid, dtype=DTYPE)

XR = tf.convert_to_tensor(XR, dtype=DTYPE)


model = tf.keras.Sequential()
# Input Layer
model.add(tf.keras.layers.InputLayer(input_shape=(layers[0])))

# Hidden Layers
for n_i in range(1, len(layers) - 1):
    model.add(tf.keras.layers.Dense(units=layers[n_i],
                                    activation='tanh',
                                    kernel_initializer='glorot_normal')
              )

model.add(tf.keras.layers.Dense(units=layers[-1], kernel_initializer='glorot_normal')
          )

solver = HRPINN(model=model, x_r=x_valid, is_pinn=False)

solver.alpha = keff / (rho * cp)
solver.beta = 4 * heff / (rho * cp * d)
solver.Ts = Ts
# Choose step sizes aka learning rate
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 5000], [1e-3, 1e-4, 1e-5])

# Solve with Adam optimizer
optim = tf.keras.optimizers.Adam()

# Start timer
solver.solve_with_tf_optimizer(optim, x_train, y_train, n_step=10000)
solver.is_pinn = True
solver.solve_with_tf_optimizer(optim, x_train, y_train, n_step=10000)

# solver.solve_with_scipy_optimizer(X_u_train, u_train, method='L-BFGS-B')
solver.solve_with_scipy_optimizer(x_train, y_train, method='SLSQP')

Temp = Temp*(Temp_ub-Temp_lb)+Temp_lb
Exact = Exact*(Temp_ub-Temp_lb)+Temp_lb
Qh = Qh*(Qh_ub-Qh_lb) + Qh_lb

yp = model(XR)
# yp = yp.numpy().reshape((n_t, n_x))

# plt.plot(yp[:, 84])
yp = yp*(Temp_ub-Temp_lb)+Temp_lb
yp = np.reshape(yp, (200, 100))

plt.figure()
tm = t / 60.0 * t_ub
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

figure, ax = plt.subplots(nrows=2, ncols=1)
plt.ion()
plot1 = ax[0].contourf(np.array([yp[0], yp[0]]), 100, cmap='jet')
plot2 = ax[1].contourf(np.array([Exact[0], Exact[0]]), 100, cmap='jet')
# plt.clim(vmin=20, vmax=50)
plt.show(block=False)
for i in range(1, len(yp), 10):
    # plot1.data = np.array([yp[i], yp[i]])
    ax[0].contourf(np.array([yp[i], yp[i]]), 100, vmin=35, vmax=45, cmap='jet')
    ax[0].set_title('Predicted')
    ax[0].xaxis.set_ticks([])
    ax[0].yaxis.set_ticks([])

    ax[1].contourf(np.array([Exact[i], Exact[i]]), 100, vmin=35, vmax=45, cmap='jet')
    ax[1].set_title('Measured')
    ax[1].xaxis.set_ticks([])
    ax[1].yaxis.set_ticks([])
    # plt.show(block=False)
    figure.canvas.draw()
    figure.canvas.flush_events()
    time.sleep(0.001)


# tail = 20
#
# figure, ax = plt.subplots()
# plt.ion()
# plot1 = ax.plot(tm[0:11], yp[0:11, -1], '-.', label='pred')[0]
# plot2 = ax.plot(tm[0:10], Exact[0:10, -1], '-', label='meas')[0]
# plt.ylim([20, 50])
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel(r'$^\circ C$')
# plt.show(block=False)
#
# for i in range(1, len(yp)-tail):
#     # plot1.data = np.array([yp[i], yp[i]])
#     plot1.set_data(tm[max(0, i-tail):i+11], yp[max(0, i-tail):i+11, -1])
#     plot2.set_data(tm[max(0, i-tail):i+10], Exact[max(0, i-tail):i+10, -1])
#     # plot2.set_ydata()
#     # plt.show(block=False)
#     ax.relim()
#     ax.autoscale_view(scaley=False)
#     figure.canvas.draw()
#     figure.canvas.flush_events()
#     time.sleep(0.1)
