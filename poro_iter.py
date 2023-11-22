
from fenics import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Parameters

dt=0.0001 
m_ = Constant(0.3) # flux on y=1
m_0 = Constant(0) # flux on x=0 and x=1
Nu=Constant(0.3) # poisson
Gc = Constant(0.005) # fracture toughness
l = 0.03

tol=1e-3 #tolerance for g(d)


"""
Create mesh and function space
"""
mesh = UnitSquareMesh(100, 100)
# mesh=RectangleMesh(Point(0, 0), Point(1, 5), 100, 100)
V_element = VectorElement('CG', mesh.ufl_cell(), 2)
Q_element = FiniteElement('CG', mesh.ufl_cell(), 1)
Mixed_element = MixedElement([V_element,Q_element])
V = FunctionSpace(mesh, Mixed_element) #Space for u and p
B = FunctionSpace(mesh, 'CG', 1) # Space for damage parameter d
WW = FunctionSpace(mesh, 'DG', 0)


"""
Functions
"""

U = Function(V)
U_i=Function(V)
u, p = split(U)

U_n = Function(V)
u_n, p_n = split(U_n)
v, q = TestFunctions(V)

d = Function(B)
b = TestFunction(B)


d_0 = Constant(0) 
d_i = interpolate(d_0, B) #initial condition for d
'''
Stain, stress, g(d), H
'''
def lambda_(Nu):    #lame constant with poisson ratio
    return Nu/((1+Nu)*(1-2*Nu))
def mu(Nu):    ##lame constant with poisson ratio
    return 1/(2*(1+Nu))
def epsilon(u):
    return 0.5*(grad(u)+grad(u).T)
def sigma(u):
    return  lambda_(Nu)*div(u)*Identity(len(u)) + 2*mu(Nu)*epsilon(u)
def g(d,tol):
    return (1-tol)*((1-d)**2)+tol
def psi(u):
    return inner(sigma(u),epsilon(u))
# def psi(u):
#     return 0.5*(lambda_(Nu)+2/3*mu(Nu))*(0.5*(tr(epsilon(u))+abs(tr(epsilon(u)))))**2+mu(Nu)*inner(dev(epsilon(u)),dev(epsilon(u)))
def H(unew,Hold):
    return conditional(lt(Hold,psi(unew)),psi(unew),Hold) #if Hold less than psi(new): H=psi(unew), else: H=Hold

# def H(uold,unew,Hold):
#     return conditional(lt(psi(uold),psi(unew)),psi(unew),Hold)
Hold = Function(WW)
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
n = FacetNormal(mesh)

"""
Weak form
"""
F1 = (div(u)*q*dx-div(u_n)*q*dx
      + dt*m_*q*ds(2)
      + dt*m_0*q*ds(1)
      + dt*dot(grad(p), grad(q))*dx) 


F2 = (g(d_i,tol)*inner(sigma(u),epsilon(v))*dx - div(v)*p*dx)

F=F1+F2


# F3 = (Gc/l*d*b*dx+Gc*l*inner(grad(d),grad(b))*dx
#       - 2*(1-tol)*(1-d)*H(u_n,u,Hold)*b*dx)
F3 = (Gc/l*d*b*dx+Gc*l*inner(grad(d),grad(b))*dx
      - 2*(1-tol)*(1-d)*H(u,Hold)*b*dx)

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
u, p = U.split()
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

    def value_shape(self):
        return (3,)
U.interpolate(InitialConditions())
U_n.assign(U)

ufile_pvd = File("poro_con_4/displacement.pvd")
pfile_pvd = File("poro_con_4/pressure.pvd")
crack_pvd = File ("poro_con_4/crack.pvd")

t=0
ufile_pvd << (u,t)
pfile_pvd << (p,t)
crack_pvd << (d,t)
"""
Solve the PDEs
"""

max_iter=50
tol_new=1e-7 #tolerance for iteration
max_iter_real=0
while t<=1:
    t += dt 
    iter = 0
    err = 1 #initialize the iteration counter and the error
    step=(t/dt) % 100

    while err > tol_new:
        iter += 1
        solver1.solve() #compute u and p
        solver2.solve() #compute d
        err_phi = errornorm(d,d_i,norm_type = 'l2',mesh = None) #compute the error in d by comparing with the 
                                                                   #previous solution by means of the L2 norm                                                    
        
        err =err_phi
        print(err)
        if err <= tol_new: 
            print(err)
            print ('Iterations:', iter, ', Total time', t)
            if iter>=max_iter_real: #record the maximum iteration during the process
                max_iter_real=iter
            # d_n.assign(d)
            U_n.assign(U)
            Hold.assign(project(H(u,Hold), WW)) 
            if abs(step-1)<1e-7:
                crack_pvd << (d,t)
                ufile_pvd << (u,t)
                pfile_pvd << (p,t)
        d_i.assign(d)
        if iter>=max_iter:
            break
    if iter>=max_iter:
        print('inconvergence')
        break
print(t)
print(max_iter_real)
print('Finished')
# plt.show()