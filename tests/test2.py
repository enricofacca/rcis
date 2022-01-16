import os
import sys
import time as cputiming
from copy import deepcopy as cp


# leave these lines to work without installation
sys.path.append(os.path.abspath("../src/"))

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

    def Save(self, file):
        """
        Procedure for saving to file

        Args:
            file (str) :: file path where save x solution
        """
        f = open(file, "w")
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
        #: b (float): Parabola linear coefficient :math: `b x`
        self.b = b
        #: lower_bound (float): Lower bound for solution
        self.lower_bound = lb
        #: upper_bound (float): Upper bound for solution
        self.upper_bound = up

    def energy(self, x):
        """
        The functional we are minimizing

        Args:
            x (real) :: position

        Return:
        value (real) :: :math: `y=a \frac{x^2}{2} + bx`
        """
        return self.a * x ** 2 + self.b * x


class GradientDescentControls:
    """
    Class with gradient decent controls.

    """

    def __init__(self, step0=0.1, verbose=0):
        """
        Control parameters of gradient descent solver
        """
        # step length
        self.step = step0

        # info solver application
        self.verbose = 0


class InfoDescent:
    """
    Class to store info of the solver
    """

    def __init__(self):
        #: cpu_gradient (real) :: cpu to compute gradient
        self.cpu_gradient = 0.0

        #: current iteration
        self.iteration = 0


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
        if ctrl is None:
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
        if unknows.x >= problem.lower_bound and unknows.x <= problem.upper_bound:
            ierr = 0
        else:
            ierr = -1

        return self, unknows, ierr

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
        if unknows.x >= problem.lower_bound and unknows.x <= problem.upper_bound:
            ierr = 0
        else:
            ierr = -1

        # Here we simulate that an error occured
        if self.infos.iteration % 4 == 0:
            ierr = 0
        if ierr == 0:
            self.infos.iteration += 1
        print("inner iter", self.infos.iteration)

        return unknows, ierr, self


def test_main(verbose=0):
    # init solution container and copy it
    sol = Real(1)

    # init inputs data
    data = Parabola(0.5)

    # init solver
    ctrl = GradientDescentControls(0.1)
    grad_desc = ParabolaDescent(ctrl)

    # init update cycle controls
    flags = CycleControls(100, verbose=verbose)

    # Extra controls to tune Gradient Descent controls
    # with an increasening time step
    min_step = 0.01
    max_step = 2
    step_expansion = 1.05
    step_contraction = 2

    # list to store
    sol_old = cp(sol)
    hystory = []
    while flags.flag >= 0:
        """
        Call reverse communication.
        Then select action according to flag.flag and flag.info
        """
        flags, sol, grad_desc = flags.new_reverse(grad_desc, data, sol)
        print("flag =", flags.flag)

        if flags.flag == 1:
            """Here the user evalutes if system reached convergence and
            and eventaully break the cycle"""
            var = abs(sol.x - sol_old.x) / grad_desc.ctrl.step
            if var < 1e-4:
                flags.flag = -1
                flags.info = 0

        if flags.flag == 2:
            """Here the user can study anything combaining solver, problem, and
            unknows.  Here we store the time, cpu, anf functional value"""
            energy = data.energy(sol.x)
            error = abs(sol.x - data.b / data.a)
            hystory.append(
                [
                    sol.time,
                    cp(grad_desc.infos),
                    energy,
                    error,
                ]
            )
            if verbose:
                print(
                    "iter:",
                    flags.iterations,
                    "energy=",
                    energy,
                    "error",
                    abs(sol.x - data.b / data.a),
                )

        if flags.flag == 3:
            """Here user have to set solver controls for next update"""
            grad_desc.ctrl.step = max(
                min(grad_desc.ctrl.step * step_expansion, max_step), min_step
            )
            # We copy data before update
            sol_old.x = cp(sol.x)

        if flags.flag == 4:
            """Un error occured. Here user have to reset solver controls after
            update failure. We restore the start solution"""
            grad_desc.ctrl.step = max(
                min(grad_desc.ctrl.step / step_contraction, max_step), min_step
            )
            print("Shrinking time step", grad_desc.ctrl.step)

            sol = cp(sol_old)

        if flags.flag == 5:
            """Here user have to settle poblem inputs Nothing to do for this
            problem since inputs do not change along the iterations"""
            pass

    """ Unzip history for plotting"""
    times, infos, energies, errors = zip(*hystory)

    if verbose:
        import matplotlib.pyplot as plt

        plt.xlabel("t")
        plt.semilogy(times, errors)
        plt.ylabel("err")
        plt.show()

    assert abs(sol.x) < 1e-3
    return 0


if __name__ == "__main__":
    sys.exit(test_main(0))
