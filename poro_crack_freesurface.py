
from fenics import *
import numpy as np
import matplotlib.pyplot as plt


# Parameters

dt=0.0001 
m_ = Constant(0.3) # flux on y=1
m_0 = Constant(0) # flux on x=0 and x=1
Nu=Constant(0.3) # poisson
Gc = Constant(0.05) # fracture toughness
l = 0.03

tol=1e-3 
"""
Create mesh and function space
"""
mesh = UnitSquareMesh(100, 100)
V_element = VectorElement('CG', mesh.ufl_cell(), 2)
Q_element = FiniteElement('CG', mesh.ufl_cell(), 1)
D_element = FiniteElement('CG', mesh.ufl_cell(), 1) # mesh for the scalar field damage parameter d
Mixed_element = MixedElement([V_element,Q_element,D_element])
V = FunctionSpace(mesh,Mixed_element)

"""
Functions
"""

U = Function(V)
u, p, d = split(U)

U_n = Function(V)
u_n, p_n, d_n = U_n.split(U_n)

v, q ,b = TestFunctions(V)


'''
Boundary condition
'''


# u=(0,0) on x=0 and x=1
def boundary_D1(x, on_boundary):
    return on_boundary and  (near(x[0], 0) or near(x[0], 1))

bc2 = DirichletBC(V.sub(0), Constant((0,0)), boundary_D1)

# p=0 on y=0
def boundary_D(x, on_boundary):
    return on_boundary and  near(x[1], 0)
bc = DirichletBC(V.sub(1), Constant((0)), boundary_D)

bcu=[bc,bc2]

# Neumann boundary on x=0 and x=1 for pressure
class NeumannBoundary_RL(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0) or near(x[0], 1))
# Neumann boundary on y=1 for pressure
class NeumannBoundary_Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1)

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundary_markers.set_all(0)
n_rlb = NeumannBoundary_RL()
n_rlb.mark(boundary_markers, 1)

n_top = NeumannBoundary_Top()
n_top.mark(boundary_markers, 2)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

'''
Stain, stress, g(d)
'''
def epsilon(u):
    return 0.5*(grad(u)+grad(u).T)
def sigma(u):
    return  Nu/((1+Nu)*(1-2*Nu))*div(u)*Identity(len(u)) + 1/(1+Nu)*epsilon(u)
def g(d,tol):
    return (1-tol)*((1-d)**2)+tol

"""
Weak form
"""
F1 = (div(u)*q*dx-div(u_n)*q*dx
      + dt*m_*q*ds(2)
      + dt*m_0*q*ds(1)
      + dt*dot(grad(p), grad(q))*dx) 


F2 = (g(d,tol)*inner(sigma(u),epsilon(v))*dx - div(v)*p*dx)

F3 = (Gc/l*d*b*dx+Gc*l*inner(grad(d),grad(b))*dx
      - 2*(1-tol)*(1-d)*inner(sigma(u),epsilon(u))*b*dx)


F=F1+F2+F3

J = derivative(F, U)


"""
Set of the nonlinear solver
"""

# set up the nonlinear problem and solver
problem = NonlinearVariationalProblem(F, U, bcu, J = J)
solver = NonlinearVariationalSolver(problem)

# customise some of the solver parameters
solver.parameters['nonlinear_solver'] = 'snes'
solver.parameters['snes_solver']['relative_tolerance'] = 1e-7
solver.parameters['snes_solver']['linear_solver'] = 'mumps'

# redefine the solution components
u, p, d = U.split()

"""
Set up the initial conditions
"""

# Here we assume that phi = 0 + random numbers and mu = 0

class InitialConditions(UserExpression):
    
    def eval(self, values, x):
        """
        This method actually sets the values for the initia conditions
        """
        # set the initial condition for u at each point
        values[0] = 0
        values[1] = 0
        # set the initial condition for p at each point
        values[2] = 0
        # set the initial condition for d at each point
        values[3] = 0

    def value_shape(self):
        return (4,)
    

# assign the initial conditions to U and U_n
U.interpolate(InitialConditions())
U_n.assign(U)

ufile_pvd = File("poro_freesurface_6/displacement.pvd")
pfile_pvd = File("poro_freesurface_6/pressure.pvd")
crack_pvd = File ("poro_freesurface_6/crack.pvd")
# conc_f = File ("ResultsDir/phi.pvd")
"""
Solve the PDEs
"""


t=0
while t<=1:
    t += dt
    solver.solve()
    u, p, d = U.split(deepcopy=True)
    # ufile_pvd << (u,t)
    # pfile_pvd << (p,t)
    step=(t/dt) % 100
    # print(step)
    if abs(step-1)<1e-7:
        crack_pvd << (d,t)
        ufile_pvd << (u,t)
        pfile_pvd << (p,t)
    u_n.assign(u)
    print(t)
   
    
print('Finished')
# plt.show()