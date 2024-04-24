import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib 
import matplotlib.animation as animation


#x-y-grid:
Lx= 10
Ly =10
Nx = 100
Ny = 100
x = np.linspace(0,Lx,Nx)
y = np.linspace(0, Ly, Ny)
X_Y_Grid = np.meshgrid(x,y)


#t-grid:
Nt = 10
Lt = 10


t = np.linspace(0, Lt, Nt)

'''#inisialiserer
def f(x_var):
    return np.sin(np.pi*x_var)
def g(y):
    return np.sin(np.pi*y)

initx = np.zeros(Nx)

initx = f(x)
inity = g(y)
'''

k=0.1
h=0.1
d=0.1
gamma=k/h**2
sigma= k/d**2
randkrav_x = np.zeros((Nx,Nt))
randkrav_y = np.zeros((Ny,Nt))


U = np.zeros((Nx, Ny, Nt))
U[50,50,0]=10
#U[4,5,0]=10
#U[5,4,0]=10
#U[4,4,0]=10
#U[:,0 , 0] = initx
#U[0,:,0] = inity
U[0, :, :] = randkrav_x
U[:,0,:] = randkrav_y

'''print(randkrav_x)

for j in range(Ny):
    for i in range(Nx):
        U[i,j,0]=initx[i]
        U[j,i,0]+=inity[j]
for i in range(Nt):
    for j in range(Nx):
        U[0,j, i]=randkrav_x[j,i]
for i in range(Nt):
    for j in range(Ny):
        U[j,0,i]=randkrav_y[j,i]'''









'''def U_lapX(x):
    -np.pi**2*np.sin(np.pi*x)
def U_lapY(y):
    -np.pi**2*np.sin(np.pi*y)
'''

for i in range(Nt-1):
    for j in range(Ny-1):
        for k in range(Nx-1):

            U[k,j,i+1]= (U[k+1,j,i]- 2*U[k,j,i]-U[k-1,j,i])*gamma + (U[k,j+1,i] -2*U[k,j,i] -U[k,j-1,i])*sigma + U[k,j,i]


'''
writer = animation.FFMpegWriter(fps=15)

fig, ax = plt.subplots(3,3,subplot_kw=dict(projection='3d'))
#fig, axs = plt.subplots(
#    3, 3, figsize=(9, 9), layout="constrained", gridspec_kw={"hspace": 0.1})
plt.show()'''
'''plt.xlim(0,10)
plt.ylim(0,10)

X, Y = np.meshgrid(x,y)
with writer.saving(fig, "exp3d.mp4", 100):

    for tval in np.linspace(0,20,160):
        print(tval)
        zval = U(x,y,rlist, tval)
        ax.set_zlim(-1, 1)
        ax.plot_surface(xlist,ylist,zval,cmap=cm.viridis)

        writer.grab_frame()
        plt.cla()'''


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update_plot(frame_number, Z, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X_Y_Grid[0], X_Y_Grid[1], Z[:, :, frame_number], cmap=cm.coolwarm)

plot = [ax.plot_surface(X_Y_Grid[0], X_Y_Grid[1], U[:, :, 0], cmap=cm.coolwarm)]

ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_zlim(0, 10)  

ani = animation.FuncAnimation(fig, update_plot, Nt, fargs=(U, plot), interval=100)
plt.show()






'''

X, Y = np.meshgrid(x,y)
ax = plt.figure().add_subplot(projection='3d')
# Plot the surface.

surf = ax.plot_surface(x, y, U[:,:,i], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.figure(figsize=(20, 20))
for t_step in range(Nt):
    plt.subplot(3, 4, t_step + 1)
    plt.contourf(X, Y, U[:, :, t_step], cmap='coolwarm')
    plt.colorbar(label='Temperature')
    plt.title(f'Time = {t[t_step]:.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
plt.tight_layout()
plt.show()'''





