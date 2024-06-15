import argparse
import symbolic


def pddl_test(pddl_path, verbose=True):
    # Load pddl
    pddl = symbolic.Pddl(
        f"{pddl_path}/domain.pddl",
        f"{pddl_path}/problem.pddl"
    )

    # Validate pddl
    is_valid = pddl.is_valid(verbose=verbose)
    print('This PDDL problem can be solved!: ', is_valid)
    if not is_valid:
        raise RuntimeError()

    # Initialize basic forward planner from the PDDL initial state.
    planner = symbolic.Planner(pddl)
    bfs = symbolic.BreadthFirstSearch(planner.root, max_depth=15, verbose=verbose)
    
    # Perform BFS until the first valid plan.
    print("Planning...")
    plan = next(iter(bfs))

    # Extract list of action to execute.
    # The first nodes in plans returned by BFS just contain the initial state (no action).
    action_skeleton = [node.action for node in plan[1:]]

    # Execute plan.
    print("Executing plan...")
    s = pddl.initial_state
    for a in action_skeleton:
        s = pddl.next_state(s, a)
    print(f"Final state: {s}\n")
    print(f"Is goal satisfied? {pddl.is_goal_satisfied(s)}\n")
    
    # Find all valid plans.
    print("Planning...")
    for idx_plan, plan in enumerate(bfs):
        print(f"Solution {idx_plan}")
        print("===========")

        # Iterate over all nodes in the plan.
        for node in plan:
            print(node)

    print('############')
    print("It's over!!!")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pddl_path", type=str, default=None, help="Pddl path.")
    parser.add_argument("--verbose", type=bool, default=True, help="Print log details.")
    args = parser.parse_args()
    pddl_test(args.pddl_path, args.verbose)
