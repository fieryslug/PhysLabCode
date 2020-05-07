import json
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from pprint import pprint

def r1():
    with open('res.json', 'r') as f:
        res = json.load(f)['1']

    x = res['x']
    V = res['V']
    plt.scatter(x, V, s=0.5)
    
    p1 = curve_fit(test_func_1, x, V, p0=[1, 7])
    xx = np.linspace(min(x), max(x), 100)
    plt.plot(xx, test_func_1(xx, -0.551208, 8.1), label='2d')

    p2 = curve_fit(test_func_2, x, V, p0=[1, 7])
    plt.plot(xx, test_func_2(xx, 1.6056, 8.1), label="3d")
    plt.grid()
    print(p2)

    plt.legend()
    plt.ylabel('V(V)')
    plt.xlabel('r(cm)')
    plt.title('V - r plot')
    plt.show()

def test_func_1(x, C, a):
    return C*(np.log(np.abs(x-a)) - np.log(np.abs(x+a)))


def test_func_2(x, C, a):
    return C*(1/(np.abs(x-a)) - 1/(np.abs(x+a)))

def r2_plot():
    theta = np.linspace(0, np.pi * 2, 1000)
    C0 = -0.551208
    V0 = [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]
    a = 8.1
    for V in V0:
        if V == 0:
            V = 1e-5
        alpha = V / C0
        b = (np.exp(2*alpha)+1) / (np.exp(2*alpha)-1) * a
        R = np.sqrt(b**2 - a**2)
        plt.plot(b+R*np.cos(theta),R*np.sin(theta), label=V)

    plt.legend()
    plt.grid()
    plt.xlabel('x(cm)')
    plt.ylabel('y(cm)')
    plt.title('equipotential lines')
    plt.show()

def r3_plot():
    theta = [0, np.pi/4, np.pi/2]
    theta_n = ['0', '\\frac{{\\pi}}{{4}}', '\\frac{{\\pi}}{{2}}']
    a = 8.1
    dt = 0.05
    C0 = -0.551208
    for i in range(len(theta)):
        th = theta[i]
        x = -a
        y = 0
        x += np.cos(th) * dt 
        y += np.sin(th) * dt 
        t = 0
        xx = [x]
        yy = [y]
        while x < a:
            t += dt
            rp = np.sqrt((x+a)**2 + y**2)
            rm = np.sqrt((x-a)**2 + y**2)
            Ex = -C0*((x+a)/rp**2 - (x-a)/rm**2)
            Ey = -C0*(y/rp**2 - y/(rm**2))
            x += Ex * dt
            y += Ey * dt
            xx.append(x)
            yy.append(y)
    
        plt.plot(xx, yy, label='$' + theta_n[i] + '$')
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.xlabel('x(cm)')
    plt.ylabel('y(cm)')
    plt.title('field lines')
    plt.show()

r3_plot()