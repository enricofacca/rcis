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
    def iterate(self, problem, unknows):
        """
        Abstract interface for iterate procedure.

        Args:
            problem: Any class describing a problem.
            unknows: Any class describing a problem unknows.

        Returns:
            self: We return the solver to get statistics
            unknows: Updated solution
            ierr (int): Error flag. 
                ierr == 0 no error occured.
                ierr != 0 unknows may not be accurate.        
        """
        return self, unknows, ierr
    
class CycleControls():
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
    def __init__(self,
                 max_iterations=1000,
                 max_restarts=10,
                 verbose=0):
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
        #: Second user communication to request for solver controls
        #: 1: controls next update 2: controls after failed update
        self.request = 0
        
        #: int: State comunication flag
        self.ierr = 0
        #: int: Iterations counter
        self.iterations = 0
        #: int: Restart counter
        self.restarts= 0
 
        """      
        Statistics of iterative algorithm
        """
        #: float: Cpu conter
        self.cpu_time = 0
        
    def reverse_communication_solver(self, solver, problem, unknows):
        """ 
        Subroutine to run reverse communition approach 
        of iterative solver.
        
        Args:
            solver (Solver): Class with iterate method
            problem: Problem description
            unknows: Problem unknows

        Returns:
            self (CycleControls): Returning changed class.
                Counters and statistics are changed.
            solver (Solver): Class modified with statistics
                             of application.
            unknows: Updated solution.
        """
        
        if (self.flag == 0):
            # study system
            self.flag = 3
            self.request = 0
            self.ierr = 0
            
            return self, unknows, solver
            
        if (self.flag == 1):
            # user has settled problem inputs and controls
            # now we update
            self.ierr = 0
            
            # Update cycle. 
            # If it succees goes to the evaluation of
            # system varaition (flag ==4 ).
            # In case of failure, reset controls
            # and ask new problem inputs or ask the user to
            # reset controls (flag == 2 + ierr=-1 ).
        
            if (self.restarts == 0):
                if (self.verbose >= 1):
                    print(' ')
                    print('UPDATE ' + str(self.iterations+1))
                cpu_update = 0.0
            else:
                print('UPDATE ' +
                      str(self.iterations + 1) +
                      ' | RESTART = ' +
                      str(self.restarts))
            #
            # update unknows
            #
            start_time = cputiming.time()
            [unknows, ierr, solver] = solver.iterate(problem, unknows)
            cpu_update += cputiming.time() - start_time
        
            #
            # check for successfulls update
            #
            if (ierr == 0):
                # info for succesfull update
                self.iterations += 1
                self.cpu_time += cputiming.time() - start_time
                if (self.verbose >= 1):                    
                    if (self.restarts == 0):
                        print(
                            'UPDATE SUCCEED CPU = ' + 
                            '{:.2f}'.format(cpu_update)
                        )
                    else:
                        print(
                            'UPDATE SUCCEED ' +
                            str(self.restarts) +
                            ' RESTARTS CPU =' +
                            '{:.2f}'.format(cpu_update)
                        )
                        print(' ')
                #
                # Ask to the user to evalute if stop cycling 
                # 
                self.flag = 2
                self.request = 1
                self.ierr = 0 
            else:
                #
                # update failed
                #
                print('UPDATE FAILURE')
            
                # try one restart more
                self.restarts += 1
            
                # stop if number max restart update is passed 
                if (self.restarts >= self.max_restarts):
                    self.flag = -1  # breaking cycle
                    self.request = 0
                    self.ierr = ierr
                else:
                    # ask the user to reset controls and problem inputs
                    self.flag = 1
                    self.request = 2
                    self.ierr = ierr
            return self, unknows, solver
       
        if (self.flag == 2):
            # check if maximum number iterations was achieded 
            if (self.iterations >= self.max_iterations):
                self.flag = -1
                if (self.verbose >= 1):
                    print(
                        'Update Number exceed limits' + 
                        str(self.max_iterations)
                    )
                # break cycle
                return self, unknows, solver

            # Let the use study the system
            self.nrestarts = 0  # we reset the count of restart
            self.flag = 3
            self.ierr = 0
            return self, unknows, solver

        if (self.flag == 3):
            # ask user new problem inputs in order to update 
            self.flag = 1
            self.request = 1
            self.ierr = 0
            return self, unknows, solver

class ConstrainedSolver(ABC): 
    """
    Abstract procedure describing the solver syncronizing i.e.,
    action of the solver on the problem unknows so they fit 
    into potential problem constraints. 
    
    Args:
    problem : class describing problem setup
    unknows : class describing problem solution

    Returns:
    self    : solver class. We return the solver
              to read statistic.
    unknows : updated solution after one 
    ierr    : integer ==0 if no error occurred.
              Values different from 0 are used to
              identifity the error occured
    
    """
    @abstractmethod
    def syncronize(self, problem, unknows):
        return unknows, ierr, self  
