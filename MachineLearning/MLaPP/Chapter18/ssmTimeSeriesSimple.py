import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
seed works different in python from in matlab, so I can't generate the perfectly match figure as in book
'''

np.random.seed(2)

def local_level(N, a0, Q, R):
    y = np.zeros(N)
    a = np.zeros(N)
    noise_a = ss.norm(0, np.sqrt(Q))
    noise_y = ss.norm(0, np.sqrt(R))
    for i in range(N):
        if i == 0:
            pre_a = a0
        else:
            pre_a = a[i - 1]

        ai = pre_a + noise_a.rvs(1)
        yi = ai + noise_y.rvs(1)
        a[i] = ai
        y[i] = yi

    return y

def local_trend(N, a0, b0, Qa, Qb, R):
    y = np.zeros(N)
    a = np.zeros(N)
    b = np.zeros(N)
    noise_a = ss.norm(0, np.sqrt(Qa))
    noise_b = ss.norm(0, np.sqrt(Qb))
    noise_y = ss.norm(0, np.sqrt(R))
    for i in range(N):
        if i == 0:
            pre_a = a0
            pre_b = b0
        else:
            pre_a, pre_b = a[i - 1], b[i - 1]

        bi = pre_b + noise_b.rvs(1)
        ai = pre_a + pre_b + noise_a.rvs(1)
        yi = ai + bi + noise_y.rvs(1)   # while in books, it lacks of bi, it's a mistake
        a[i] = ai
        b[i] = bi
        y[i] = yi

    return y

def seasonal(N, A, C, Q, R, init_mu):
    '''
    use matrix dot to represent seasonal model, refer to ldsSample.m
    %   x(t+1) = A * x(t) + G*u(t) + w(t),  w ~ N(0, Q),  x(0) = init_state
    %   y(t) =   C * x(t) + v(t),  v ~ N(0, R)
    x means all the hidden variables, it's a column vector, including a, b, c1, c2, c3..etc

    e.g :
    A = [1  1  0  0  0
         0  1  0  0  0
         0  0 -1 -1  -1
         0  0  1  0  0
         0  0  0  1  0];

    C = [1  1  1  0  0];

    then:
      a(t+1) = at + bt + w(t)
      b(t+1) = bt + w(t)
      c1(t+1) = -c1 - c2 - c3 + w(t)
      c2(t+1) = c1 + w(t)
      c3(t+1) = c2 + w(t)

      y(t) = at + bt + c1(t)
    '''
    initial = init_mu.reshape(-1, 1)
    D = len(initial)
    C = C.reshape(1, -1)
    states = np.zeros((N, D))
    y = np.zeros(N)
    mu = np.zeros(D)
    print(Q, D)
    noise_state = ss.multivariate_normal(mu, np.sqrt(Q) * np.eye(D), allow_singular=True)
    noise_obs = ss.norm(0, np.sqrt(R))
    for i in range(N):
        if i == 0:
            pre_state = initial
        else:
            pre_state = states[i - 1].reshape(-1, 1)

        state = np.dot(A, pre_state).ravel() + noise_state.rvs(1)
        state = state.reshape(-1, 1)
        yi = np.dot(C, state) + noise_obs.rvs(1)
        states[i] = state.ravel()
        y[i] = yi

    return y

# local level
N = 200
a0 = 1
Qs = [0, 0.1, 0.1]
Rs = [0.1, 0, 0.1]
xs = np.linspace(1, N, N)
ys = []    # for local levels
for i in range(len(Qs)):
    ys.append(local_level(N, a0, Qs[i], Rs[i]))

# plot local level
fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('ssmTimeSeriesSimple')

ax1 = plt.subplot(131)
ax1.tick_params(direction='in')
plt.axis([0, 200, -8, 6])
plt.xticks(np.linspace(0, 200, 11))
plt.yticks(np.linspace(-8, 6, 8))
plt.title('local level, a=1.000')
plt.plot(xs, ys[0], 'k-', lw=1.5, label='Q=0.0, R=0.1')
plt.plot(xs, ys[1], 'r:', lw=1.5, label='Q=0.1, R=0.0')
plt.plot(xs, ys[2], 'b-.', lw=1.5, label='Q=0.1, R=0.1')
plt.legend()

# local linear trend
np.random.seed(11)
N2 = 100
a2, b2 = 10, 1
Qs2 = [0, 1.0, 1.0]
Rs2 = [100.0, 0, 100.0]
xs2 = np.linspace(1, N2, N2)
ys2 = []    # for local trends
for i in range(len(Qs2)):
    ys2.append(local_trend(N2, a2, b2, Qs2[i], Qs2[i], Rs2[i]))

# plot local trend
ax2 = plt.subplot(132)
ax2.tick_params(direction='in')
plt.axis([0, 100, -800, 200])
plt.xticks(np.linspace(0, 100, 11))
plt.yticks(np.linspace(-800, 200, 6))
plt.title('local trend, a=10.000, b=1.000')
plt.plot(xs2, ys2[0], 'k-', label='Q=0.0, R=100.0')
plt.plot(xs2, ys2[1], 'r:', label='Q=1.0, R=0.0')
plt.plot(xs2, ys2[2], 'b-.', label='Q=1.0, R=100.0')
plt.legend()

# seasonal model
np.random.seed(0)
N3 = 20
A = np.array([[1, 1, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, -1, -1, -1],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0]])
C = np.array([1, 1, 1, 0, 0])
Qs3 = [0, 1, 1]
Rs3 = [1, 0, 1]
init_mu = np.array([0, 0, 1, 1, 1])
xs3 = np.linspace(1, N3, N3)
ys3 = []
for i in range(len(Qs3)):
    ys3.append(seasonal(N3, A, C, Qs3[i], Rs3[i], init_mu))

ax3 = plt.subplot(133)
ax3.tick_params(direction='in')
# plt.axis([0, 100, -800, 200])
plt.xticks(np.linspace(0, 20, 5))
# plt.yticks(np.linspace(-800, 200, 6))
plt.title('seasonal model, s=4, a=0.000, b=0.000')
plt.plot(xs3, ys3[0], 'k-', label='Q=0.0, R=1.0')
plt.plot(xs3, ys3[1], 'r:', label='Q=1.0, R=0.0')
plt.plot(xs3, ys3[2], 'b-.', label='Q=1.0, R=1.0')
plt.legend()

plt.tight_layout()
plt.show()