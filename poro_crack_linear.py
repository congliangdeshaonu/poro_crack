
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

tol=1e-3 

"""
Create mesh and function space
"""
mesh = UnitSquareMesh(100, 100)
V_element = VectorElement('CG', mesh.ufl_cell(), 2)
Q_element = FiniteElement('CG', mesh.ufl_cell(), 1)
Mixed_element = MixedElement([V_element,Q_element])
V = FunctionSpace(mesh, Mixed_element) #Space for u and p
B = FunctionSpace(mesh, 'CG', 1) # Space for the scalar field damage parameter d
WW = FunctionSpace(mesh, 'DG', 0)


"""
Functions
"""

U = Function(V)
u, p = split(U)

U_n = Function(V)
u_n, p_n = U_n.split()
v, q = TestFunctions(V)
D= Function(B)
d, b= TrialFunction(B), TestFunction(B)


d_0 = Constant(0)
d_n = interpolate(d_0, B)
'''
Stain, stress, g(d)
'''
def epsilon(u):
    return 0.5*(grad(u)+grad(u).T)
def sigma(u):
    return  Nu/((1+Nu)*(1-2*Nu))*div(u)*Identity(len(u)) + 1/(1+Nu)*epsilon(u)
def g(d,tol):
    return (1-tol)*((1-d)**2)+tol
def psi(u):
    return inner(sigma(u),epsilon(u))
def H(unew,Hold):
    return conditional(lt(Hold,psi(unew)),psi(unew),Hold) #if Hold less than psi(new): H=psi(unew), else: H=Hold


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

bcu=[bc,bc2] #boundary condition for u and p
bcd=[] #boundary condition for d

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
dnew,Hold = Function(B), Function(B)

"""
Weak form
"""
F1 = (div(u)*q*dx-div(u_n)*q*dx
      + dt*m_*q*ds(2)
      + dt*m_0*q*ds(1)
      + dt*dot(grad(p), grad(q))*dx) 


F2 = (g(d_n,tol)*inner(sigma(u),epsilon(v))*dx - div(v)*p*dx)

F=F1+F2


F3 = (Gc/l*d*b*dx+Gc*l*inner(grad(d),grad(b))*dx
      - 2*(1-tol)*(1-d)*H(u,Hold)*b*dx)

J1 = derivative(F, U)

n = FacetNormal(mesh)

"""
Set of the nonlinear solver
"""

# set up the nonlinear problem and solver
problem1 = NonlinearVariationalProblem(F, U, bcu, J = J1)
solver1 = NonlinearVariationalSolver(problem1)


problem2=LinearVariationalProblem(lhs(F3), rhs(F3), dnew, bcd)

# customise some of the solver parameters
solver1.parameters['nonlinear_solver'] = 'snes'
solver1.parameters['snes_solver']['relative_tolerance'] = 1e-7
solver1.parameters['snes_solver']['linear_solver'] = 'mumps'

solver2= LinearVariationalSolver(problem2)


# redefine the solution components
u, p = U.split()


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


ufile_pvd = File("poro_H_36/displacement.pvd")
pfile_pvd = File("poro_H_36/pressure.pvd")
crack_pvd = File ("poro_H_36/crack.pvd")
# conc_f = File ("ResultsDir/phi.pvd")
t=0
ufile_pvd << (u,t)
pfile_pvd << (p,t)
crack_pvd << (dnew,t)
"""
Solve the PDEs
"""



while t<=0.6:
    t += dt
    iter = 0
    err = 1 #initialize the iteration counter and the error
    step=(t/dt) % 100
    while err > tol:
        iter += 1
        solver1.solve() #compute u and p
        u, p = U.split(deepcopy=True)
        u_n.assign(u) #assign u
        solver2.solve() #compute d
        err_phi = errornorm(dnew,d_n,norm_type = 'l2',mesh = None) #compute the error in d by comparing with the 
                                                                   #previous solution by means of the L2 norm

        Hold.assign(project(H(u,Hold), WW)) 
        d_n.assign(dnew)
        err =err_phi
        if err < tol:
            print ('Iterations:', iter, ', Total time', t)
            if abs(step-1)<1e-7:
                crack_pvd << (dnew,t)
                ufile_pvd << (u,t)
                pfile_pvd << (p,t)
        if iter>=10:
            break
    if iter>=10:
        print('inconvergence')
        break
        
    # print(t)
    
   
    
# print('Finished')
# plt.show()
