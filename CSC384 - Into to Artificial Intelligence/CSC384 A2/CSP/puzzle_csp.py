#Look for #IMPLEMENT tags in this file.
'''
All models need to return a CSP object, and a list of lists of Variable objects 
representing the board. The returned list of lists is used to access the 
solution. 

For example, after these three lines of code

    csp, var_array = caged_csp_model(board)
    solver = BT(csp)
    solver.bt_search(prop_FC, var_ord)

var_array[0][0].get_assigned_value() should be the correct value in the top left
cell of the FunPuzz puzzle.

The grid-only models do not need to encode the cage constraints.

1. binary_ne_grid (worth 10/100 marks)
    - A model of a FunPuzz grid (without cage constraints) built using only 
      binary not-equal constraints for both the row and column constraints.

2. nary_ad_grid (worth 10/100 marks)
    - A model of a FunPuzz grid (without cage constraints) built using only n-ary 
      all-different constraints for both the row and column constraints. 

3. caged_csp_model (worth 25/100 marks) 
    - A model built using your choice of (1) binary binary not-equal, or (2) 
      n-ary all-different constraints for the grid.
    - Together with FunPuzz cage constraints.

'''
from cspbase import *
import itertools


def binary_ne_grid(fpuzz_grid):
    dom = []
    for i in range(fpuzz_grid[0][0]):
        dom.append(i + 1)

    vars = []
    for i in dom:
        vars_r = []
        for j in dom:
            vars_r.append(Variable('Q{}{}'.format(i, j), dom))
        vars.append(vars_r)

    sat_tuples = []
    for t in itertools.product(dom, dom):
        if t[0] != t[1]:
            sat_tuples.append(t)

    csp = CSP("{}Binary_ne".format(fpuzz_grid[0][0]*fpuzz_grid[0][0]), list(itertools.chain(*vars)))

    for qi in range(len(dom)):
        for qj in range(len(dom)):
            for qk in range(1, (len(dom))):
                if qj < qk:
                    con_row = Constraint("R(Q{}{},Q{}{})".format(qi + 1, qj + 1, qi + 1, qk + 1),
                                         [vars[qi][qj], vars[qi][qk]])
                    con_row.add_satisfying_tuples(sat_tuples)
                    csp.add_constraint(con_row)
                if qi < qk:
                    con_column = Constraint("C(Q{}{},Q{}{})".format(qi + 1, qj + 1, qk + 1, qj + 1),
                                            [vars[qi][qj], vars[qk][qj]])
                    con_column.add_satisfying_tuples(sat_tuples)
                    csp.add_constraint(con_column)

    return csp, vars
    

def nary_ad_grid(fpuzz_grid):
    dom = []
    for i in range(fpuzz_grid[0][0]):
        dom.append(i + 1)

    vars = []
    for i in dom:
        vars_r = []
        for j in dom:
            vars_r.append(Variable('Q{}{}'.format(i, j), dom))
        vars.append(vars_r)

    sat_tuples = []
    for t in itertools.permutations(dom):
        sat_tuples.append(t)

    csp = CSP("{}Nary_ad".format(fpuzz_grid[0][0] * fpuzz_grid[0][0]), list(itertools.chain(*vars)))

    for qi in range(len(dom)):
        con_row = Constraint('R{}'.format(qi+1), vars[qi])
        con_column = Constraint('C{}'.format(qi+1), [v[qi] for v in vars])
        con_row.add_satisfying_tuples(sat_tuples)
        csp.add_constraint(con_row)
        con_column.add_satisfying_tuples(sat_tuples)
        csp.add_constraint(con_column)

    return csp, vars
    

def caged_csp_model(fpuzz_grid):
    csp, vars = binary_ne_grid(fpuzz_grid)

    dom = []
    for i in range(fpuzz_grid[0][0]):
        dom.append(i + 1)

    for cage_i in range(1, len(fpuzz_grid)):
        cage = fpuzz_grid[cage_i]
        if len(cage) > 2:
            op = cage.pop(-1)
            tar = cage.pop(-1)
            cage_vars = []
            cage_dom = []
            for c in cage:
                cage_vars.append(vars[c // 10 - 1][c % 10 - 1])
                cage_dom.append(vars[c // 10 - 1][c % 10 - 1].domain())
            con = Constraint('C{}'.format(cage_i), cage_vars)
            sat_tuples = []
            if len(cage_vars) < 1:
                tuples = itertools.product(dom)
            else:
                tuples = itertools.product(dom, repeat=len(cage_vars))
            for t in tuples:
                if op == 0 and sum(t) == tar and t not in sat_tuples:
                    sat_tuples.append(t)
                elif op == 1 and t not in sat_tuples:
                    for p in itertools.permutations(t):
                        if (p[0]-sum(p[1:])) == tar:
                            sat_tuples.append(t)
                elif op == 2 and t not in sat_tuples:
                    for p in itertools.permutations(t):
                        res = p[0]
                        for p_i in p[1:]:
                            res /= p_i
                        if res == tar:
                            sat_tuples.append(t)
                elif op == 3 and t not in sat_tuples:
                    res = 1
                    for t_i in t:
                        res *= t_i
                    if res == tar:
                        sat_tuples.append(t)
            con.add_satisfying_tuples(sat_tuples)
            csp.add_constraint(con)
        else:
            c = cage.pop(0)
            tar = cage.pop(0)
            con = Constraint('{}Cage'.format(cage_i), vars[c // 10 - 1][c % 10 - 1])
            con.add_satisfying_tuples(tuple([tar]))
            csp.add_constraint(con)

    return csp, vars
    