import sys
import numpy as np
from pyquest import ops
from pyquest import cheat
from pyquest import utils
from pyquest.utils import reporting


def run_example_interactive():
    """
    Running a damping example to demonstrate decoherence in pyQuEST
    """

    # creating environment
    env = utils.createQuestEnv()()
    # allocating qubit register
    qureg = utils.createDensityQureg()(1, env=env)
    cheat.initPlusState()(qureg=qureg)
    reporting.reportStateToScreen()(qureg=qureg, env=env, a=0)
    for i in range(0, 10):
        ops.applyOneQubitDampingError()(qureg=qureg, qubit=0, probability=0.1)
        reporting.reportStateToScreen()(qureg=qureg, env=env, a=0)

    qureg = utils.createDensityQureg()(1, env=env)
    cheat.initPlusState()(qureg=qureg)
    reporting.reportStateToScreen()(qureg=qureg, env=env, a=0)
    for i in range(0, 10):
        ops.applyOneQubitDepolariseError()(qureg=qureg, qubit=0, probability=0.1)
        reporting.reportStateToScreen()(qureg=qureg, env=env, a=0)


if __name__ == '__main__':
    run_example_interactive()
