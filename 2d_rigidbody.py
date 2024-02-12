
from fenics import *
# from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from mpi4py import MPI


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
# Parameters

# Parameters

dt=0.01
m_ = Constant(1) # flux on y=1
m_0 = Constant(0) # flux on x=0 and x=1     
Nu=Constant(0.3) # poisson
Gc = Constant(0.001) # fracture toughness
l = 0.01
lambda_=Nu/((1+Nu)*(1-2*Nu))    #lame constant with poisson ratio
mu=1/(2*(1+Nu)) ##lame constant with poisson ratio
tol=1e-3 #tolerance for g(d)
tol_new=1e-3 #tolerance for iteration

"""
Create mesh and function space
"""

class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] 
        y[1] = x[1]-1.0
pbc = PeriodicBoundary()


mesh = RectangleMesh(Point(0, 0), Point(1, 1), 100, 100)
# mesh_name = 'com_9'
# mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(0.1, 1, 0.1),5, 10, 5)

"""
    mesh, functions, etc
"""
# mesh = Mesh()
# hdf = HDF5File(mesh.mpi_comm(), 'mesh/mesh_compare/'+mesh_name+'.h5', "r")
# hdf.read(mesh, "/mesh", False)
# subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
# hdf.read(subdomains, "/subdomains")
# bdry = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
# hdf.read(bdry, "/boundaries")

V_element = VectorElement('CG', mesh.ufl_cell(), 2)
Q_element = FiniteElement('CG', mesh.ufl_cell(), 1)
R_element = FiniteElement("R", mesh.ufl_cell(), 0)  # Lagrange multipliers space for nullspace enforcement
Mixed_element = MixedElement([V_element,Q_element,R_element,R_element,R_element])
V = FunctionSpace(mesh, Mixed_element,constrained_domain=pbc) #Space for u and p
B = FunctionSpace(mesh, 'CG', 1) # Space for damage parameter d
WW = FunctionSpace(mesh, 'DG', 0)
S = TensorFunctionSpace(mesh, 'P', 1)






"""
Functions
"""

U = Function(V)
U_i=Function(V)
u, p, ph1, ph2, ph3= split(U)

U_n = Function(V)
u_n, p_n, pn1, pn2, pn3= split(U_n)
v, q, qh1, qh2, qh3 = TestFunctions(V)

d = Function(B)
b = TestFunction(B)


d_0 = Constant(0) 
d_i = interpolate(d_0, B) #initial condition for d
'''
Stain, stress, g(d), H
'''

sigma_i = Function(S)
def epsilon(u):
    return 0.5*(grad(u)+grad(u).T)
def sigma(u):
    return  lambda_*div(u)*Identity(len(u)) + 2*mu*epsilon(u)
def g(d,tol):
    return (1-tol)*((1-d)**2)+tol
def hist(u,p):
    return 0.5*inner(sigma(u),epsilon(u))-p*div(u)
# def hist(u,p):
    # return -p*div(u)
def H(unew,p,Hold):
    return conditional(lt(Hold,hist(unew,p)),hist(unew,p),Hold) #if Hold less than psi(new): H=psi(unew), else: H=Hold
Hold = Function(WW)
'''
Boundary condition
'''


# p=0 on x=1
def boundary_D(x, on_boundary):
    return on_boundary and  near(x[0], 1)
bc = DirichletBC(V.sub(1), Constant((0)), boundary_D)

bcu=[bc]


# Neumann boundary on y=0 and y=1 for pressure
class NeumannBoundary_RL(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0) or near(x[1], 1))

# Neumann boundary on x=0 for pressure
class NeumannBoundary_Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundary_markers.set_all(0)
n_rlb = NeumannBoundary_RL()
n_rlb.mark(boundary_markers, 1)

n_top = NeumannBoundary_Top()
n_top.mark(boundary_markers, 2)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
n = FacetNormal(mesh)

"""
Weak form
"""
F1 = (div(u)*q*dx-div(u_n)*q*dx
      + dt*m_*q*ds(2)
      + dt*m_0*q*ds(1)
      + dt*dot(grad(p), grad(q))*dx) 


F2=inner(g(d_i,tol)*(sigma(u)-p*Identity(len(u))),grad(v))*dx

F=F1+F2

# Translations
Z_transl = [Constant((1, 0)), Constant((0, 1))]
# Rotations
Z_rot = [Expression(('x[1]', '-x[0]'), degree=1)]

# All
Z = Z_transl + Z_rot

for i, Zi in enumerate(Z):
    F -= locals()[f"ph{i+1}"] * inner(v, Zi) * dx
    F -= locals()[f"qh{i+1}"] * inner(u, Zi) * dx


F3 = (Gc/l*d*b*dx+Gc*l*inner(grad(d),grad(b))*dx
      - 2*(1-tol)*(1-d)*H(u,p,Hold)*b*dx)




J1 = derivative(F, U)
J3 = derivative(F3, d)


"""
Set of the nonlinear solver
"""

# set up the nonlinear problem and solver
problem1 = NonlinearVariationalProblem(F, U, bcu, J = J1)
solver1 = NonlinearVariationalSolver(problem1)



problem2 = NonlinearVariationalProblem(F3, d, bcs=[],J = J3)
solver2 = NonlinearVariationalSolver(problem2)


# customise some of the solver parameters
solver1.parameters['nonlinear_solver'] = 'snes'
solver1.parameters['snes_solver']['relative_tolerance'] = 1e-7
solver1.parameters['snes_solver']['linear_solver'] = 'mumps'




solver2.parameters['nonlinear_solver'] = 'snes'
solver2.parameters['snes_solver']['relative_tolerance'] = 1e-7
solver2.parameters['snes_solver']['linear_solver'] = 'mumps'

# redefine the solution components
u, p, ph1, ph2, ph3 = U.split()
#

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
        values[3] = 0
        values[4] = 0
        values[5] = 0

    def value_shape(self):
        return (6,)
U.interpolate(InitialConditions())
U_n.assign(U)


folder_name = "rigid_body_1"
output_dir = f"../results/{folder_name}"
if rank == 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)


ufile_pvd = File(f"{output_dir}/displacement.pvd")
pfile_pvd = File(f"{output_dir}/pressure.pvd")
crack_pvd = File(f"{output_dir}/crack.pvd")
sigma_pvd = File(f"{output_dir}/stress.pvd")





t=0
ufile_pvd << (u,t)
pfile_pvd << (p,t)
crack_pvd << (d,t)
sigma_pvd << (sigma_i,t)
"""
Solve the PDEs
"""

max_iter=50

max_iter_real=0

while t<=1:
    t += dt 
    iter = 0
    err = 1 #initialize the iteration counter and the error
    step=(t/dt) % 10
    solver1.solve() #compute u and p
    U_n.assign(U)
    Hold.assign(project(H(u,p,Hold), WW))
    solver2.solve() #compute d
    d_i.assign(d)
    print ('Total time', t)    
    computed_stress = sigma(u)
    sigma_i.assign(project(computed_stress, S))
    # if abs(step-1)<1e-7:
    crack_pvd << (d,t)
    ufile_pvd << (u,t)
    pfile_pvd << (p,t)
    sigma_pvd << (sigma_i,t)


print('Finished')
