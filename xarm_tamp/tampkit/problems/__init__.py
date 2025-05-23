from xarm_tamp.tampkit.problems.assembly.assembly_problem import (
    fmb_momo_problem, siemense_gearbox_problem, peg_in_hole_problem,
    block_world_problem,
)
from xarm_tamp.tampkit.problems.stacking.stacking_problem import stacking_problem
from xarm_tamp.tampkit.problems.fetch.fetch_problem import fetch_problem

PROBLEMS = [
    fmb_momo_problem,
    siemense_gearbox_problem,
    stacking_problem,
    fetch_problem,
    peg_in_hole_problem,
    block_world_problem
]