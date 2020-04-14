from hepatitis_decision_tree import main as run
import sys

for i in range(10):
    sys.argv = ['hepatitis-decision-tree.py', f'hepatitis-training-run-{i}',f'hepatitis-test-run-{i}']
    print("\n\trun",i)
    run()