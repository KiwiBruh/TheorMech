import numpy as np
from math import pi
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from scipy.integrate import odeint


def odesys(y, t, m1, m2, c, c1, R, g): 
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = -m2 * R * np.cos(y[1]) 
    a12 = ((2 * m1 + m2) * R**2 + m2 * y[0] + 2 * R * m2 * y[0] * np.sin(y[1])) 
    a21 = 1 
    a22 = -R * np.cos(y[1]) 

    b1 = -m2 * R * y[0] * y[3]**2 * np.cos(y[1]) - 2 * m2 * (y[0] + R * np.sin(y[1])) * y[2] * y[3] - c1 * R**2 * y[1] - m2 * g * y[0] * np.cos(y[1]) 
    b2 = y[0] * y[3]**2 - 2 * (c/m2) * y[0] - g * np.sin(y[1]) 
    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21) 
    dy[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21) 

    return dy


# константы
t_fin = 20
t = np.linspace(0, t_fin, 1001) 

x0 = 5 # положение кольца в начальный момент времени

m1 = 2 # масса кольца
R = 3 # радиус кольца

m2 = 1 # масса бруска

c = 5 # коэффециент упругости пружин в кольце
c1 = 5 # коэффециент упругости горизонтальной пружины

g = 9.81 # ускорение свободного падения

phi0 = 0 # поворот кольца в начальный момент времени
dphi0 = 0 # угловая скорость кольца в начальный момент времени

s0 = 0 # отклонение груза в начальный момент времени
ds0 = 0 # скорость отклонения груза в начальный момент времени


y0 = [s0, phi0, ds0, dphi0]
Y = odeint(odesys, y0, t, (m1, m2, c, c1, R,  g))

s = Y[:, 0] # отклонение груза
ds = Y[:, 2] # скорость отклонения груза
dds = [odesys(y, time, m1, m2, c, c1, R, g)[2] for y, time in zip(Y, t)] # ускорение отклонения груза

phi = Y[:, 1] # угол поворота кольцы
dphi = Y[:, 3] # угловая скорость кольца
ddphi = [odesys(y, time, m1, m2, c, c1, R, g)[3] for y, time in zip(Y, t)] # угловое ускорение колца

angles = np.linspace(0, 2 * pi , 360)

box_w = 0.4 # ширина груза
box_h = 0.2  # высота груза


# генерирует пружинку высотой h с количеством витков k и шириной w
def spring(k, h, w):
    x = np.linspace(0, h, 100)
    return np.array([
        x,
        np.sin(2 * pi / (h / k) * x) * w
    ])


# массивы, где будем хранить просчитанные точки
ring_dots_x_tmp = R * np.cos(angles)
ring_dots_y_tmp = R * np.sin(angles)

box_x_tmp = np.array([-box_h / 2, -box_h / 2, box_h / 2, box_h / 2, -box_h / 2])
box_y_tmp = np.array([-box_w / 2, box_w / 2, box_w / 2, -box_w / 2, -box_w / 2])

ring_dots_x = np.zeros([len(t), len(angles)])
ring_dots_y = np.zeros([len(t), len(angles)])

box_dots_x = np.zeros([len(t), 5])
box_dots_y = np.zeros([len(t), 5])

spring_a_x = np.zeros([len(t), 100])
spring_a_y = np.zeros([len(t), 100])

spring_b_x = np.zeros([len(t), 100])
spring_b_y = np.zeros([len(t), 100])

spring_c_x = np.zeros([len(t), 100])
spring_c_y = np.zeros([len(t), 100])


# считаем точки
for i in range(len(t)):
    # центр кольца
    ring_x = x0 + phi[i] * R
    ring_y = R

    # точки окружности кольца
    ring_dots_x[i] = np.cos(phi[i]) * ring_dots_x_tmp + np.sin(phi[i]) * ring_dots_y_tmp + ring_x
    ring_dots_y[i] = - np.sin(phi[i]) * ring_dots_x_tmp + np.cos(phi[i]) * ring_dots_y_tmp + ring_y

    # точки груза
    bx = box_x_tmp - s[i]
    by = box_y_tmp
    box_dots_x[i] = np.cos(phi[i]) * bx + np.sin(phi[i]) * by + ring_x
    box_dots_y[i] = - np.sin(phi[i]) * bx + np.cos(phi[i]) * by + ring_y

    # горизонтальная пружина
    spring_a_x[i] = spring(5, ring_x, 0.2)[0]
    spring_a_y[i] = spring(5, ring_x, 0.2)[1] + ring_y

    # верхняя пружина в кольце
    b_x = R - spring(10, R + s[i] - box_h / 2, 0.2)[0]
    b_y = spring(10, R - s[i], 0.2)[1]
    spring_b_x[i] = np.cos(phi[i]) * b_x + np.sin(phi[i]) * b_y + ring_x
    spring_b_y[i] = -np.sin(phi[i]) * b_x + np.cos(phi[i]) * b_y + ring_y

    # нижняя пружина в кольце
    c_x = spring(10, R - s[i] - box_h / 2, 0.2)[0] - R
    c_y = spring(10, R - s[i], 0.2)[1]
    spring_c_x[i] = np.cos(phi[i]) * c_x + np.sin(phi[i]) * c_y + ring_x
    spring_c_y[i] = -np.sin(phi[i]) * c_x + np.cos(phi[i]) * c_y + ring_y


fig_for_graphs = plt.figure(figsize=[13,7]) 


# график отклонения груза
ax_for_graphs = fig_for_graphs.add_subplot(2, 3,1) 
ax_for_graphs.plot(t, s, color='blue') 
ax_for_graphs.set_title("s(t)") 
ax_for_graphs.set(xlim=[0, t_fin]) 
ax_for_graphs.grid(True)


# график скорости отклонения груза
ax_for_graphs = fig_for_graphs.add_subplot(2, 3, 2)
ax_for_graphs.plot(t,ds,color='blue')
ax_for_graphs.set_title("s'(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)


# график ускорения отклонения груза
ax_for_graphs = fig_for_graphs.add_subplot(2, 3, 3)
ax_for_graphs.plot(t,dds,color='blue')
ax_for_graphs.set_title("s''(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)


# график угла поворота кольца
ax_for_graphs = fig_for_graphs.add_subplot(2, 3, 4) 
ax_for_graphs.plot(t,phi,color='red')
ax_for_graphs.set_title('phi(t)')
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True) 


# график угловой скорости кольца
ax_for_graphs = fig_for_graphs.add_subplot(2, 3, 5) 
ax_for_graphs.plot(t,dphi,color='red')
ax_for_graphs.set_title("phi'(t)")
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True) 


# график углового ускорения кольца
ax_for_graphs = fig_for_graphs.add_subplot(2, 3, 6) 
ax_for_graphs.plot(t,ddphi,color='red')
ax_for_graphs.set_title("phi''(t)")
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True) 



# рисуем график
fig = plt.figure() # создаем холст, на котором будем рисовать фигуры
ax = fig.add_subplot(1, 1, 1)
ax.axis("equal")

surface = ax.plot([0, 0, 10], [10, 0, 0]) # пол и стена
ring, = ax.plot(ring_dots_x[0], ring_dots_y[0]) # кольцо
box, = ax.plot(box_dots_x[0], box_dots_y[0]) # груз
spring_a, = ax.plot(spring_a_x[0], spring_a_y[0]) # горизонтальная пружина
spring_b, = ax.plot(spring_b_x[0], spring_b_y[0]) # фиолетовая пружина
spring_c, = ax.plot(spring_c_x[0], spring_c_y[0]) # коричневая пружина


# функция анимации
def animate(i):
    ring.set_data(ring_dots_x[i], ring_dots_y[i])
    box.set_data(box_dots_x[i], box_dots_y[i])
    spring_a.set_data(spring_a_x[i], spring_a_y[i])
    spring_b.set_data(spring_b_x[i], spring_b_y[i])
    spring_c.set_data(spring_c_x[i], spring_c_y[i])

    return ring, box, spring_a, spring_b, spring_c


animation = FuncAnimation(fig, animate, frames=len(t), interval=60)
plt.show()
