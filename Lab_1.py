import numpy as np
def matrix_ops(A):
    b=A.T
    print(np.dot(A,b))


def main():
    a=np.array([[1,2,3],[4,5,6]])
    print(matrix_ops(a))

if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
def plot_map1(x):
    y=(2*x)+3
    for i in x:
        plt.plot(x,y)
    plt.show()

def main():
    b=np.linspace(-100,100,100)
    print(plot_map1(b))

if __name__ == '__main__':
    main()

def plot_map2(x):
    y=(2*(x**2))+3*x+4
    for i in x:
        plt.plot(x,y)
    plt.show()

def main():
    b=np.linspace(-10,10,100)
    print(plot_map2(b))

if __name__ == '__main__':
    main()

def plot_map2(x):
    y=(1/(15*np.sqrt((2*(np.pi))))*(2.71828)**(-0.5*((x-0/15))**2))
    for i in x:
        plt.plot(x,y)
    plt.show()

def main():
    b=np.linspace(-100,100,100)
    print(plot_map2(b))

if __name__ == '__main__':
    main()

def plot_map3(x):
    y=x**2
    z=2*x
    for i in x:
        plt.plot(x,y)
        plt.plot(x,z,marker='*')
    plt.show()

def main():
    b=np.linspace(-100,100,100)
    print(plot_map3(b))

if __name__ == '__main__':
    main()

def plot_map3(x):
    y=x**2
    z=2*x
    for i in x:
        plt.plot(x,y,)
        plt.plot(x,z,marker='*',ms=8,mfc='red')
    plt.show()

def main():
    b=np.linspace(-10,10,10)
    print(plot_map3(b))

if __name__ == '__main__':
    main()
