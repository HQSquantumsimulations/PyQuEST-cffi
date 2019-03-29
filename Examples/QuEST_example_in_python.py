import sys
import numpy as np
from pyquest_cffi import ops
from pyquest_cffi import cheat
from pyquest_cffi import utils
from pyquest_cffi.utils import reporting


def run_example_interactive():
    """
    Running the exact same Example QuEST provides in the QuEST git repository
    with the interactive python interface of PyQuEST-cffi
    """
    print('PyQuEST-cffi tutorial based on QuEST tutorial')
    print('     Basic 3 qubit circuit')

    # creating environment
    env = utils.createQuestEnv()()
    # allocating qubit register
    qureg = utils.createQureg()(3, env=env)
    cheat.initZeroState()(qureg=qureg)

    # Using the report function to print system status
    print('This is the environment:')
    reporting.reportQuESTEnv()(env=env)
    print('This is the qubit register:')
    reporting.reportQuregParams()(qureg=qureg)
    print('This we can easily do in interactive python:')
    print('Result of qureg.isDensityMatrix: ', qureg.isDensityMatrix)

    # Apply circuit

    ops.hadamard()(qureg=qureg, qubit=0)
    ops.controlledNot()(qureg=qureg, control=0, qubit=1)
    ops.rotateY()(qureg=qureg, qubit=2, theta=0.1)

    ops.multiControlledPhaseFlip()(qureg=qureg, controls=[0, 1, 2], number_controls=3)

    u = np.zeros((2, 2), dtype=complex)
    u[0, 0] = 0.5*(1+1j)
    u[0, 1] = 0.5*(1-1j)
    u[1, 0] = 0.5*(1-1j)
    u[1, 1] = 0.5*(1+1j)
    ops.unitary()(qureg=qureg, qubit=0, matrix=u)

    a = 0.5+0.5*1j
    b = 0.5-0.5*1j
    ops.compactUnitary()(qureg=qureg, qubit=1, alpha=a, beta=b)

    v = np.array([1, 0, 0])
    ops.rotateAroundAxis()(qureg=qureg, qubit=2, theta=np.pi/2, vector=v)

    ops.controlledCompactUnitary()(qureg=qureg, control=0, qubit=1, alpha=a, beta=b)

    ops.multiControlledUnitary()(qureg=qureg,  controls=[
        0, 1], number_controls=2, qubit=2, matrix=u)

    # cheated results

    print('Circuit output')

    print('Probability amplitude of |111> by knowing the index: ',
          cheat.getProbAmp()(qureg=qureg, index=7))
    print('Probability amplitude of |111> by referencing basis state: ',
          cheat.getProbAmp()(qureg=qureg, index=[1, 1, 1]))

    # measuring:
    measurement = ops.measure()(qureg=qureg, qubit=0)
    print('Qubit 0 was measured as: ', measurement)


if __name__ == '__main__':
    run_example_interactive()
