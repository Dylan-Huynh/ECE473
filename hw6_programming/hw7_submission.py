import argparse
import datetime
import json
import random
import signal
import itertools
import copy
import numpy as np


class TimeoutException(Exception):
    pass


def handle_maxSeconds(signum, frame):
    raise TimeoutException()


VERBOSE = True


# return True if clause has a true literal
def check_clause(clause, assignment):
    clause_val = False
    for i in clause:
        if np.sign(i) == np.sign(assignment[abs(i) - 1]):  # assignment is 0-ref, variables 1-ref
            clause_val = True
    return clause_val


def check(clauses, assignment):
    global VERBOSE

    if VERBOSE:
        print('Checking assignment {}'.format(assignment))
        print('score of assignment is {}'.format(score(clauses, assignment)))
    for clause in clauses:
        if not check_clause(clause, assignment):
            return clause
    print('Check succeeded!')
    return True


def backtrack_search(num_variables, clauses):
    print('Backtracking search started')

    def backtrack(assignment, i):
        # i is next variable number to try values for (1..numvariables)
        if i == num_variables + 1:  # no more variables to try
            if check(clauses, assignment) == True:
                return assignment
            return None
        else:
            for val in [-1, 1]:
                assignment[i - 1] = val  # assignment is 0-ref, so assignment[x] stores variable x+1 value
                result = backtrack(assignment, i + 1)
                if result != None:
                    return result
        return None

    assignment = np.array([1] * num_variables)
    result = backtrack(assignment, 1)
    print('Backtracking search completed successfully')
    return (result)


def random_walk(num_variables, clauses):
    print('Random walk search started')
    assignment = np.ones(num_variables)
    while True:
        if True == check(clauses, assignment):
            break
        var_to_flip = random.randint(1, num_variables)
        assignment[var_to_flip - 1] *= -1
    print('Random walk search completed successfully')
    return assignment


def generate_solvable_problem(num_variables):
    global VERBOSE

    k = 3  # 3-SAT
    random.seed()

    clauses_per_variable = 4.2  # < 4.2 easy;  >4.2 usually unsolvable.  4.2 challenging to determine.
    num_clauses = round(clauses_per_variable * num_variables)

    # this assignment will solve the problem
    target = np.array([2 * random.randint(0, 1) - 1 for _ in range(num_variables)])
    clauses = []
    for i in range(num_clauses):
        seeking = True
        while seeking:
            clause = sorted((random.sample(range(0, num_variables), k)))  # choose k variables at random
            clause = [i + 1 for i in clause]
            clause = [(2 * random.randint(0, 1) - 1) * clause[x] for x in range(k)]  # choose their signs at random
            seeking = not check_clause(clause, target)
        clauses.append(clause)

    if VERBOSE:
        print('Problem is {}'.format(clauses))
        print('One solution is {} which checks to {}'.format(target, check(clauses, target)))

    return clauses


class Solver:
    def __init__(self, cnf, num_var):
        cnf_l = list(cnf)
        self.cnf = []
        for item in cnf_l:
            if item not in self.cnf:
                self.cnf.append(list(item))
        self.literals = []
        for i in range(num_var):
            self.literals.append(i)
        self.num_var = num_var

    def solve(self):
        return self.dpll(self.num_var, self.cnf, self.literals)

    def dpll(self, num_variables, clauses, assignment=None):

        # print('clauses', clauses, 'len', len(clauses), 'assignment', assignment)

        if assignment is None:
            assignment = np.array([1] * num_variables)

        while sum(len(clause) == 1 for clause in clauses):  # repeat until there are no unit clauses

            for clause in clauses:

                if len(clause) == 1:
                    if clause[0] > 0:
                        assignment[clause[0] - 1] = 1
                        clauses = self.unit_p(clause[0], clauses)
                        break
                    else:
                        assignment[-clause[0] - 1] = -1
                        clauses = self.unit_p(clause[0], clauses)
                        break

        for clause in clauses:
            if len(clause) == 0:
                return None

        if len(clauses) == 0:
            return assignment

        tau = abs(clauses[0][0])

        assignment_true = assignment.copy()
        assignment_false = assignment.copy()
        clauses_true = copy.deepcopy(clauses)
        clauses_false = copy.deepcopy(clauses)

        if clauses[0][0] > 0:
            assignment_true[tau - 1] = 1
            assignment_false[tau - 1] = -1
        else:
            assignment_true[tau - 1] = -1
            assignment_false[tau - 1] = 1

        clauses_true = self.unit_p(clauses_true[0][0], clauses_true)
        clauses_false = self.unit_p(-clauses_false[0][0], clauses_false)

        # print("tau to try is:", clauses[0][0])
        try1 = self.dpll(num_variables, clauses_true, assignment_true)
        if try1 is not None:
            assignment = try1
            return assignment
        # print('try 1 failed')

        # print("tau to try is:", -clauses[0][0])
        try2 = self.dpll(num_variables, clauses_false, assignment_false)
        if try2 is not None:
            assignment = try2
            return assignment
        else:
            assignment = None

        return assignment

    def unit_p(self, alpha, clauses_arr):
        # alpha is pos or neg variable
        clauses_arr = [x for x in clauses_arr if alpha not in x]  # delete clauses containing alpha
        for x in clauses_arr:
            if -alpha in x:  # remove !alpha from all clauses
                x.remove(-alpha)
        return clauses_arr

    def most_common(self, lst):
        merged = list(itertools.chain(*lst))  # most frequent item
        if len(merged) > 0:
            return max(set(merged), key=merged.count)
        else:
            self.ou = True  # length of senetence is 0

def hw7_submission(num_variables, clauses, timeout):  # timeout is provided in case your method wants to know
    s = Solver(clauses, num_variables)
    result = s.solve()
    if result is None:
        return False
    return result

'''
def dpll(num_variables, clauses, result=None):
    def unit_prop(l, clauses_arr):
        # alpha is pos or neg variable
        clauses_arr = [x for x in clauses_arr if l not in x]  # delete clauses containing alpha
        for x in clauses_arr:
            if -l in x:  # remove !alpha from all clauses
                x.remove(-l)
        return clauses_arr

    def pure_lit(l, clauses):

        return clauses

    if result is None:
        result = [1] * num_variables

    while True:  # repeat until there are no unit clauses
        for clause in clauses:

            if len(clause) == 1:  # find a unit clause
                # print('UNIT CLAUSE FOUND', clause)
                if clause[0] > 0:  # if unit clause is "True"
                    result[clause[0] - 1] = 1
                    clauses = unit_prop(clause[0], clauses)
                    break
                else:  # if unit clause is "False"
                    result[-clause[0] - 1] = -1
                    clauses = unit_prop(clause[0], clauses)
                    break
        done_check = 0
        for clause in clauses:
            if len(clause) == 1:
                done_check += 1
        if done_check == 0:
            break

    for l in range(num_variables):


    for clause in clauses:
        if len(clause) == 0:
            return None

    if len(clauses) == 0:
        # if all clauses have been removed this expression is correct
        return result

    # pick the first variable in the first clause and test it
    alpha = abs(clauses[0][0])  # get the abs of the first variable

    assignment_true = result.copy()
    assignment_false = result.copy()
    clauses_true = copy.deepcopy(clauses)
    clauses_false = copy.deepcopy(clauses)

    if clauses[0][0] > 0:  # if chosen alpha is true
        assignment_true[alpha - 1] = 1
        assignment_false[alpha - 1] = -1
    else:  # if chosen alpha is false
        assignment_true[alpha - 1] = -1
        assignment_false[alpha - 1] = 1

    clauses_true = unit_prop(clauses_true[0][0], clauses_true)
    clauses_false = unit_prop(-clauses_false[0][0], clauses_false)

    # print("alpha to try is:", clauses[0][0])
    try1 = dpll(num_variables, clauses_true, assignment_true)
    if try1 is not None:
        assignment = try1
        return assignment
    # print('try 1 failed')

    # print("alpha to try is:", -clauses[0][0])
    try2 = dpll(num_variables, clauses_false, assignment_false)
    if try2 is not None:
        assignment = try2
        return assignment
    else:
        assignment = None

    return assignment'''

def solve_SAT(file, save, timeout, num_variables, algorithms, verbose):
    global VERBOSE

    VERBOSE = verbose

    if file != None:
        with open(file, "r") as f:
            [num_variables, timeout, clauses] = json.load(f)
        print('Problem with {} variables and timeout {} seconds loaded'.format(num_variables, timeout))
    else:
        clauses = generate_solvable_problem(num_variables)
        if timeout == None:
            timeout = round(60 * num_variables / 100)
        print('Problem with {} variables generated, timeout is {}'.format(num_variables, timeout))

    if save != None:
        with open(save, "w") as f:
            json.dump((num_variables, timeout, clauses), f)

    if 'hw7_submission' in algorithms:
        signal.signal(signal.SIGALRM, handle_maxSeconds)
        signal.alarm(timeout)
        startTime = datetime.datetime.now()
        try:
            result = "Timeout"
            result = hw7_submission(num_variables, clauses, timeout)
            print('Solution found is {}'.format(result))
            if not (True == check(clauses, result)):
                print('Returned assignment incorrect')
        except TimeoutException:
            print("Timeout!")
        endTime = datetime.datetime.now()
        seconds_used = (endTime - startTime).seconds
        signal.alarm(0)
        print('Search returned in {} seconds\n'.format(seconds_used))
    if 'backtrack' in algorithms:
        signal.signal(signal.SIGALRM, handle_maxSeconds)
        signal.alarm(timeout)
        startTime = datetime.datetime.now()
        try:
            result = "Timeout"
            result = backtrack_search(num_variables, clauses)
            print('Solution found is {}'.format(result))
            if not (True == check(clauses, result)):
                print('Returned assignment incorrect')
        except TimeoutException:
            print("Timeout!")
        endTime = datetime.datetime.now()
        seconds_used = (endTime - startTime).seconds
        signal.alarm(0)
        print('Search returned in {} seconds\n'.format(seconds_used))
    if 'random_walk' in algorithms:
        signal.signal(signal.SIGALRM, handle_maxSeconds)
        signal.alarm(timeout)
        startTime = datetime.datetime.now()
        try:
            result = "Timeout"
            result = random_walk(num_variables, clauses)
            print('Solution found is {}'.format(result))
            if not (True == check(clauses, result)):
                print('Returned assignment incorrect')
        except TimeoutException:
            print("Timeout!")
        endTime = datetime.datetime.now()
        seconds_used = (endTime - startTime).seconds
        signal.alarm(0)
        print('Search returned in {} seconds\n'.format(seconds_used))


def main():
    parser = argparse.ArgumentParser(description="Run stochastic search on a 3-SAT problem")
    parser.add_argument("algorithms", nargs='*',
                        help="Algorithms to try",
                        choices=['random_walk', 'hw7_submission', 'backtrack'])
    parser.add_argument("-f", "--file", help="file name with 3-SAT formula to use", default=None)
    parser.add_argument("-s", "--save", help="file name to save problem in", default=None)
    parser.add_argument("-t", "--timeout", help="Seconds to allow (default based on # of vars)", type=int, default=None)
    parser.add_argument("-n", "--numvars", help="Number of variables (default 20)", type=int, default=20)
    parser.add_argument("-v", "--verbose", help="Whether to print tracing information", action="store_true")

    args = parser.parse_args()
    file = args.file
    save = args.save
    timeout = args.timeout
    num_variables = args.numvars
    algorithms = args.algorithms
    verbose = args.verbose

    if (file != None and (timeout != None or num_variables != None)):
        print('\nUsing input file, any command line parameters for number of variables and timeout will be ignored\n')
    solve_SAT(file, save, timeout, num_variables, algorithms, verbose)


if __name__ == '__main__':
    main()

# if you prefer to load the file rather than use command line
# parameters, use this section to configure the solver
#
# outfile = None # 'foo.txt'
# infile  = None # 'save500.txt'
# timeout = None #ignored if infile is present, will be set based on numvars if None here
# numvars = 70   #ignored if infile is present
# algorithms = ['random_walk', 'backtrack', 'hw7_submission']
# verbosity = False
# solve_SAT(infile, outfile, timeout, numvars, algorithms, verbosity)