from xarm_tamp.tampkit.problems.assembly.assembly_problem import fmb_momo_problem, fmb_simo_problem
from xarm_tamp.tampkit.problems.stacking.stacking_problem import stacking_problem
from xarm_tamp.tampkit.problems.carrying.carrying_problem import carrying_problem

PROBLEMS = [
    fmb_momo_problem,
    fmb_simo_problem,
    stacking_problem,
    carrying_problem,
]