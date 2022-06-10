import time as cputiming

# associate a number to a task
rcis_flag_break_for_error = -1 
rcis_flag_begin = 0
rcis_flag_iterate = 1
rcis_flag_check_convergence = 2
rcis_flag_study = 3
rcis_flag_reset_controls_after_failure = 4
rcis_flag_set_controls_next_update = 5
rcis_flag_set_inputs = 6


class CycleControls:
    """Class containg controls and infos to apply iterative solver.

    This class contains variables and a reverse
    communicaiton method for applying iterative a
    solver. Constructor procedure setting main controls
    (e.g., maximum iterations number, verbosity) and setting
    to zero counter variables (e.g. number of iterations
    performed)
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
        self.start_time = 0.0

    def task_description(self, flag):
        """Produce a string describing the task associated to flag
        Args:
            flag(int): Flag
        Returns:
            msg(str): Task description
        """
        if flag == rcis_flag_iterate:
            msg = "Make one solver step"
        if flag == rcis_flag_check_convergence:
            msg = "Compute if convergence is achieved. Set flag=-1 and ierr=0."
        if flag == rcis_flag_study:
            msg = "Study system."
        if flag == rcis_flag_reset_controls_after_failure:
            msg = "Set solver controls after failure."
        if flag == rcis_flag_set_controls_next_update:
            msg = "Set solver controls for next update."       
        if flag == rcis_flag_set_inputs:
            msg = "Set problem input."
        return msg


    def reverse_communication(self):
        """
        Subroutine to run reverse communition approach
        of iterative solver.

        Args:
            self (CycleControls):  Counters and statistics

        Returns:
            self (CycleControls): Returning changed class.
        """
        
        if self.flag == rcis_flag_begin:
            """Begin cycle. User can now study the system"""
            self.flag = rcis_flag_study
            self.ierr = 0
            return

        if self.flag == rcis_flag_check_convergence:
            """An iteration was completed and user checked if converge was
            achieved and decided to continue.  We check if iteration
            number exceeded the maximum. If yes we break the cycle
            passing a negative flag. Otherwise we let the user studing
            the stystem.
            """
            if self.iterations >= self.max_iterations:
                self.flag = rcis_flag_break_for_error
                if self.verbose >= 1:
                    print("Update Number exceed limits" + str(self.max_iterations))

            # we tell the user that he/she can studies the Let the use
            # study the system
            self.restarts = 0  # we reset the count of restart
            self.flag = rcis_flag_study
            self.ierr = 0
            return

        if self.flag == rcis_flag_study:
            """User studied the updated system.
            Now, we need solver controls for next update."""
            self.flag = rcis_flag_set_controls_next_update
            self.ierr = 0
            return

        if ((self.flag == rcis_flag_set_controls_next_update) or
            (self.flag == rcis_flag_reset_controls_after_failure)):
            """User set or reset solver controls.  Now, use must set new problem
            inputs, if required"""
            self.flag = rcis_flag_set_inputs
            self.ierr = 0
            return

        if self.flag == rcis_flag_set_inputs:
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
                # reset cpu time counter
                self.cpu_wasted = 0.0
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
            self.start_time = cputiming.time()
            # call update
            self.flag = rcis_flag_iterate
            return

        if (self.flag == rcis_flag_iterate):
            cpu_update = cputiming.time() - self.start_time

            # different action according to ierr
            if self.ierr == 0:
                """Succesfull update"""

                self.iterations += 1
                self.cpu_time += cputiming.time() - self.start_time
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
                self.flag = rcis_flag_check_convergence
                self.ierr = 0

            elif self.ierr > 0:
                """Update failed"""
                if self.verbose >= 1:
                    print("UPDATE FAILURE")

                # sum the cpu wasted
                self.cpu_wasted += cpu_update

                # Try one restart more
                self.restarts += 1

                # Stop if number max restart update is passed
                if self.restarts >= self.max_restarts:
                    self.flag = rcis_flag_break_for_error  # breaking cycle
                else:
                    # Ask the user to reset controls and problem inputs
                    self.flag = rcis_flag_reset_controls_after_failure
            elif self.ierr < 0:
                # Solver return negative ierr to ask more inputs
                self.flag = rcis_flag_set_inputs
            return

        def cpu_statistics(self, list_cpu_update, list_cpu_wasted):
            """Function to create a list with cpu_update and cpu_wasted by 
            each updatex.
            Call it at rcis_flag_study 
            """
            list_cpu_update.append(self.cpu_time - list_cpu_update.append[-1])
            list_cpu_wasted.append(self.cpu_wasted)
