import time as cputiming
from abc import ABC
from abc import abstractmethod


class Solver(ABC):
    """Abstract class with the **iterate** abstact method.

    This class defines the interface of **iterate**.  Any child class
    od Solver class will contain all variables, controls and procedure
    for applying **iterate**.
    """

    @abstractmethod
    def iterate(self, problem, solution):
        """
        Abstract interface for iterate procedure.

        Args:
            problem: Any class describing a problem.
            solution: Any class describing a problem solution.

        Returns:
            self: We return the solver to get statistics
            solution: Updated solution
            ierr (int): Error flag.
                ierr == 0 no error occured.
                ierr != 0 solution may not be accurate.
        """
        return self, solution, ierr


class CycleControls:
    """Class containg controls and infos to apply iterative solver.

    This class contains the variable and reverse communicaiton method
    for applying iteratively a solver to find a solution of given a
    problem. Constructor procedure setting main controls (e.g.,
    maximum iterations number, verbosity) and setting to zero counter
    variables (e.g. number of iterations performed)

    Args:
        max_iter (int) : maximum iterations number
        max_restart (int): maximum restart number
        verbose (int): verbosity
    """

    def __init__(self, max_iterations=1000, max_restarts=10, verbose=0):
        """
        Constructor of CycleControls setting main controls.

        Constructor procedure setting main controls
        (e.g., maximum iterations number, verbosity)
        and setting to zero counter variables
        (e.g. number of iterations performed)

        Args:
            max_iterations (int) : maximum iterations number
            max_restarts (int): maximum restart number
            verbose (int): verbosity level
        """

        """      
        Controls of iterative algorithm
        """
        #: int: Maximum iteration number
        self.max_iterations = max_iterations
        #: int: Maximum number of restart
        self.max_restarts = max_restarts
        #: int: Algorithm verbosity. verbose==0 silent
        #: verbose>0 more infos are printed
        self.verbose = verbose

        #: User comunication flag
        #: 1->set inputs, 2->stop criteria 3->study
        self.flag = 0

        #: int: State comunication flag
        self.ierr = 0
        #: int: Iterations counter
        self.iterations = 0
        #: int: Restart counter
        self.restarts = 0

        """      
        Statistics of iterative algorithm
        """
        #: float: Cpu conter
        self.cpu_time = 0

    def task_description(self, flag):
        """Produce a string describing the task associated to flag
        Args:
            flag(int): Flag
        Returns:
            msg(str): Task description
        """
        if flag == 1:
            msg = "Compute if convergence is achieved. Set flag=-1 and ierr=0."
        if flag == 2:
            msg = "Study system."
        if flag == 3:
            msg = "Set solver controls for next update."
        if flag == 4:
            msg = "Set solver controls after failure."
        if flag == 5:
            msg = "Set problem input."
        return msg

    def reverse_communication(self, solver, problem, solution):
        """
        Subroutine to run reverse communition approach
        of iterative solver.

        Args:
            solver (Solver): Class with iterate method
            problem: Problem description
            solution: Problem solution

        Returns:
            self (CycleControls): Returning changed class.
                Counters and statistics are changed.
            solver (Solver): Class modified with statistics
                             of application.
            solution: Updated solution.
        """

        if self.flag == 0:
            """Begin cycle. User can now study the system"""
            self.flag = 2
            self.ierr = 0
            return self, solution, solver

        if self.flag == 1:
            """An iteration was completed and user checked if converge was
            achieved and decided to continue.  We check if iteration
            number exceeded the maximum. If yes we break the cycle
            passing a negative flag. Otherwise we let the user studing
            the stystem.
            """
            if self.iterations >= self.max_iterations:
                self.flag = -1
                if self.verbose >= 1:
                    print("Update Number exceed limits" + str(self.max_iterations))
                # break cycle
                return self, solution, solver

            # we tell the user that he/she can studies the Let the use
            # study the system
            self.restarts = 0  # we reset the count of restart
            self.flag = 2
            self.ierr = 0
            return self, solution, solver

        if self.flag == 2:
            """User studied the updated system.
            Now, we need solver controls for next update."""
            self.flag = 3
            self.ierr = 0
            return self, solution, solver

        if self.flag == 4:
            """And error occured after update.  Now, user must change the solver
            controls for trying further iteration"""
            self.flag = 5
            self.ierr = 0
            return self, solution, solver

        if (self.flag == 3) or (self.flag == 5):
            """User set or reset solver controls.  Now, use must set new problem
            inputs, if required"""
            self.flag = 6
            self.ierr = 0
            return self, solution, solver

        if self.flag == 6:
            """User set/reset solver controls and problem inputs
            Now we update try to iterate"""
            self.ierr = 0

            # Update cycle.
            # If it succees goes to the evaluation of
            # system varaition (flag ==4 ).
            # In case of failure, reset controls
            # and ask new problem inputs or ask the user to
            # reset controls (flag == 2 + ierr=-1 ).

            if self.restarts == 0:
                if self.verbose >= 1:
                    print(" ")
                    print("UPDATE " + str(self.iterations + 1))
            else:
                if self.verbose >= 1:
                    print(
                        "UPDATE "
                        + str(self.iterations + 1)
                        + " | RESTART = "
                        + str(self.restarts)
                    )

            # update solution
            start_time = cputiming.time()
            [solution, ierr, solver] = solver.iterate(problem, solution)
            cpu_update = cputiming.time() - start_time

            # different action according to ierr
            if ierr == 0:
                """Succesfull update"""

                self.iterations += 1
                self.cpu_time += cputiming.time() - start_time
                if self.verbose >= 1:
                    if self.restarts == 0:
                        print("UPDATE SUCCEED CPU = " + "{:.2f}".format(cpu_update))
                    else:
                        print(
                            "UPDATE SUCCEED "
                            + str(self.restarts)
                            + " RESTARTS CPU ="
                            + "{:.2f}".format(cpu_update)
                        )
                        print(" ")

                """ Ask to the user to evalute if stop cycling """
                self.flag = 1
                self.ierr = 0

            elif ierr > 0:
                """Update failed"""
                if self.verbose >= 1:
                    print("UPDATE FAILURE")

                # Try one restart more
                self.restarts += 1

                # Stop if number max restart update is passed
                if self.restarts >= self.max_restarts:
                    self.flag = -1  # breaking cycle
                    self.ierr = ierr
                else:
                    # Ask the user to reset controls and problem inputs
                    self.flag = 4
                    self.ierr = ierr
            elif ierr < 0:
                # Solver return negative ierr to ask more inputs
                self.flag = 6
                self.ierr = ierr

            return self, solution, solver
