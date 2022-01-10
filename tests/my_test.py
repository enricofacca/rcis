import sys
import time as cputiming
from copy import deepcopy as cp



# leave these lines to work without installation
import os; sys.path.append(os.path.abspath(
'../src/')
    )
from rcis import Solver
from rcis import CycleControls

class Real:
    """
    Class describing a problem solution on real line
    """
    def __init__(self, x0):
        """
        Constructor setting initial solution
        """
        #: solution  
        self.x = x0

        #: current time  
        self.time = 0.0

        #: current iteration 
        self.iteration = 0

    def Save(self, file):
        """
        Procedure for saving to file

        Args:
            file (str) :: file path where save x solution
        """        
        f.open(file, 'w')  
        f.write(self.x)  
        f.close()  

# the r before comment allows to write Latex expression containg backslash
class Parabola:
    r"""
    Class describing a parabola.

    It contains the coeffients  a, b descringing the parabola
    :math: `y=a \frac{x^2}{2} + bx`
    
    Args:
        a (float): Parabola concavity (term :math:`a \frac{x^2}{2}`)
        b (float): Parabola linear term (term :math:`b x`)
    """    
    def __init__(self, a, b=0, lb=-1e30, up=1e30):
        """
        Constructor Parabola
        """

        #: a (float): Parabola quadratic coefficient :math:`a \frac{x^2}{2}`
        self.a = a
        #: b (float): Parabola linear coefficient :math: `b x~
        self.b = b
        #: lower_bound (float): Lower bound for solution
        self.lower_bound = lb
        #: upper_bound (float): Upper bound for solution
        self.upper_bound = up


class GradientDescentControls:
    """
    Class with gradient decent controls. 
    
    """
    def __init__(self,
                 step0=0.1,
                 verbose=0):
        """
        Control parameters of gradient descent solver
        """
        # step length
        self.step = step0

        # info solver application
        self.verbose = 0


class InfoDescent:
    def __init__(self):
        """
        Class to store info of the solver
        """
        self.cpu_gradient = 0.0

        
class ParabolaDescent(Solver): 
    """
    We extend the class "Solver"
    ovverriding the "syncronize" and the "iterate"
    procedure. 
    """
    def __init__(self, ctrl=None):
        """
        Initialize solver with passed controls (or default)
        and initialize structure to store info on solver application
        """
        # init controls
        if (ctrl is None):
            self.ctrl = GradientDescentControls(0.1)
        else:
            self.ctrl = cp(ctrl)
        # init infos
        self.infos = InfoDescent()
            
    def syncronize(self, problem, unknows):
        """
        Since the there are no constrain
        the first procedure does nothing.
        """

        # example of how the setup of our problem
        # can influence solver execution
        if (
                unknows.x >= problem.lower_bound and
                unknows.x <= problem.upper_bound
        ):
            ierr = 0
        else:
            ierr = -1
            
        return unknows, ierr, self

    def iterate(self, problem, unknows):
        """
        The update is one step of gradient descent.
        Currently with explicit euler.
        """
        start_time = cputiming.time() 
        gradient_direction = -problem.a * unknows.x
        self.infos.cpu_gradient = cputiming.time() - start_time
        
        unknows.x += self.ctrl.step * gradient_direction
        unknows.time += self.ctrl.step
        
        # example of how the setup of our problem
        # can influence solver execution
        if (
                unknows.x >= problem.lower_bound and
                unknows.x <= problem.upper_bound
        ):
            ierr = 0
        else:
            ierr = -1

        self.test = unknows.x
        
        return unknows, ierr, self


def test_main():
    # init solution container and copy it
    sol = Real(1)

    # init inputs data
    data = Parabola(0.5)

    # init solver
    ctrl = GradientDescentControls(0.1)
    grad_desc = ParabolaDescent(ctrl)

    # init update cycle controls
    flags = CycleControls(100, verbose=0)
    
    # Extra controls to tune Gradient Descent controls
    # with an increasening time step
    min_step = 0.01
    max_step = 2
    step_expansion = 1.05
    step_contraction = 2

    # list to store 
    sol_old = cp(sol)
    hystory = []
    while (flags.flag >= 0):
        """
        Call reverse communication.
        Then select action according to flag.flag and flag.info
        """
        flags, sol, grad_desc = flags.reverse_communication_solver(
            grad_desc,
            data,
            sol)
        
        # set problem inputs 
        if (flags.flag == 1):
            """
            Here user have to settle poblem inputs
            """
            #Nothing to do for this problem

            if (flags.request == 1):  
                """
                Here user have to set solver
                controls for next update
                """
                grad_desc.ctrl.step = max(min(grad_desc.ctrl.step * step_expansion,max_step),min_step)
                    

                # First we copy data before update
                sol_old.x = cp(sol.x)


            elif ( flags.request == 2):
                """
                Here user have to reset solver
                controls after update failure
                """
                grad_desc.ctrl.step = max(min(grad_desc.ctrl.step * step_expansion,max_step),min_step)
            
        if (flags.flag == 2):
            """
            Here the user evalutes if
            if convergence is achieved 
            and breaks the cycle setting 
            
            flags.flag = -1
            flags.info = 0 
            """
            var = abs(sol.x - sol_old.x) / grad_desc.ctrl.step
            
            if (var < 1e-4):
                flags.flag = -1
                flags.info = 0

                
        if (flags.flag == 3):
            """
            Here the user can study anything 
            combaining solver, problem, and unknows.
            User can save data at this point.
            """ 
            sol.iteration = cp(flags.iterations)
            hystory.append([sol.time, cp(grad_desc.infos)])
        

    assert abs(sol.x) < 1e-3
    return 0


if __name__ == "__main__":
    sys.exit(test_main())
