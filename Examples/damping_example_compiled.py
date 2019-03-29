import sys
import numpy as np
from pyquest_cffi import ops
from pyquest_cffi import cheat
from pyquest_cffi import utils
from pyquest_cffi.utils import reporting


def run_example_compiled():
    """
    Running a damping example to demonstrate decoherence in PyQuEST-cffi
    """
    lines = list()
    preamble = utils.createProgrammPreamble(interactive=False)(return_type='Complex',
                                                               function_name='tmp_QuEST_function',
                                                               arguments=[('float', 'probability')])

    lines.extend(preamble)
    lines.extend(utils.defineVariable(interactive=False)(
        vartype='Complex', name='readout', length=4, local=False))
    # creating environment
    lines.extend(utils.createQuestEnv(interactive=False)(env_name='env'))
    # allocating qubit register
    lines.extend(utils.createDensityQureg(interactive=False)
                 (num_qubits=1, env='env', qureg_name='qureg'))
    lines.extend(cheat.initPlusState(interactive=False)(qureg='qureg'))
    lines.extend(
        reporting.reportStateToScreen(interactive=False)(
            qureg='qureg', env='env'))
    for i in range(0, 10):
        lines.extend(
            ops.applyOneQubitDampingError(interactive=False)(
                qureg='qureg', qubit=0, probability='probability[0]'))
        lines.extend(reporting.reportStateToScreen(
            interactive=False)(qureg='qureg', env='env'))
    lines.extend(
        cheat.getDensityMatrix(interactive=False)(
            qureg='qureg',
            readout='readout',
            number_qubits=1))
    """
    lines.append(utils.createDensityQureg(interactive=False)(1, env='env', qureg_name='qureg2'))
    lines.append(cheat.initPlusState(interactive=False)(qureg='qureg2'))
    lines.append(reporting.reportStateToScreen(interactive=False)(qureg='qureg2', env='env'))
    for i in range(0, 10):
        lines.append(ops.applyOneQubitDepolariseError(interactive=False)
                     (qureg='qureg2', qubit=0, probability='probability[0]'))
        lines.append(reporting.reportStateToScreen(interactive=False)(qureg='qureg2', env='env'))"""

    end = utils.createProgrammEnd(interactive=False)(return_name='readout')

    lines.extend(end)

    utils.write_code_to_disk(lines=lines, file_name='tmp_QuEST_code.c')

    compiler = utils.QuESTCompiler(file_name='tmp_QuEST_code.c',
                                   code_lines=lines,
                                   return_type='Complex',
                                   function_name='tmp_QuEST_function',
                                   arguments=[('float', 'probability')])

    compiler.compile(compiled_module_name="_compiled_tmp_quest_programm")

    output = utils.callCompiledQuestProgramm(
        compiled_module_name='_compiled_tmp_quest_programm')(
            function_name='tmp_QuEST_function', length_result=4, probability=[0.1])
    print(output)


if __name__ == '__main__':
    run_example_compiled()
