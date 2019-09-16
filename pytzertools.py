from numpy import array, log10
from scipy.misc import derivative
import pytzer as pz

def solutes2arrays(solutes):
    fixions = array([sol for sol in solutes if not sol.startswith('t_')])
    fixmols = array([solutes[sol] for sol in solutes if not
        sol.startswith('t_')])
    eles = array([sol for sol in solutes if sol.startswith('t_')])
    eles.sort()
    tots = array([solutes[ele] for ele in eles])
    return fixions, fixmols, eles, tots

def printmols(fixions, fixmols, eles=[], tots=[]):
    if len(eles) > 0:
        print('Fixed solute molalities:')
    else:
        print('Solute molalities:')
    for i, ion in enumerate(fixions):
        print('{:.5f} mol/kg-H2O = [{}]'.format(fixmols[i], ion))
    if len(eles) > 0:
        print('\nEquilibrating solute total molalities:')
        for e, ele in enumerate(eles):
            print('{:.5f} mol/kg-H2O = [{}]'.format(tots[e],
                ele.split('t_')[1]))

def solve(solutearrays, tempK, pres):
    lnk_HSO4, lnk_trisH, lnk_MgOH, lnk_H2O = _getlnks(tempK)
    allmols, allions = _solvelnks(solutearrays, tempK, pres, lnk_HSO4,
        lnk_trisH, lnk_MgOH, lnk_H2O)
    return allmols, allions

def _getlnks(tempK):
    lnk_HSO4 = pz.dissociation.HSO4_CRP94(tempK)
    lnk_MgOH = pz.dissociation.MgOH_CW91(tempK)
    lnk_trisH = pz.dissociation.trisH_BH64(tempK)
    lnk_H2O = pz.dissociation.H2O_MF(tempK)
    return lnk_HSO4, lnk_trisH, lnk_MgOH, lnk_H2O

def _solvelnks(solutearrays, tempK, pres, lnk_HSO4, lnk_trisH, lnk_MgOH,
        lnk_H2O):
    eqstate_guess = [-16.5, 9.5, 0.0, 30.0]
    fixions, fixmols, eles, tots = solutearrays
    allions = pz.properties.getallions(eles, fixions)
    allmxs = pz.matrix.assemble(allions, array([tempK]), array([pres]),
        prmlib=pz.libraries.MarChemSpec)
    allmols, allions = _solve(eqstate_guess, tots, fixmols, eles, allions,
        fixions, allmxs, lnk_HSO4, lnk_trisH, lnk_MgOH, lnk_H2O)
    return allmols, allions

def _solve(eqstate_guess, tots, fixmols, eles, allions, fixions, allmxs,
        lnk_HSO4, lnk_trisH, lnk_MgOH, lnk_H2O):
    lnks = array([lnk_HSO4, lnk_MgOH, lnk_trisH, lnk_H2O])
    eqstate = pz.equilibrate.solvequick(eqstate_guess, tots, fixmols, eles,
        allions, fixions, allmxs, lnks)
    allmols, allions = pz.equilibrate.eqstate2mols(eqstate.x, tots, fixmols,
        eles, fixions)
    return allmols, allions

def _solve_pHT(solutearrays, tempK, pres, lnk_HSO4, lnk_trisH, lnk_MgOH,
        lnk_H2O):
    allmols, allions = _solvelnks(solutearrays, tempK, pres, lnk_HSO4,
        lnk_trisH, lnk_MgOH, lnk_H2O)
    mH = allmols[allions == 'H']
    mHSO4 = allmols[allions == 'HSO4']
    pH_Total = -log10(mH + mHSO4)
    return pH_Total[0]

def _pHT_grad_HSO4(solutearrays, tempK, pres, lnk_HSO4, lnk_trisH, lnk_MgOH,
        lnk_H2O):
    return derivative(lambda lnk_HSO4: _solve_pHT(solutearrays, tempK, pres,
        lnk_HSO4, lnk_trisH, lnk_MgOH, lnk_H2O), lnk_HSO4, dx=1e-6)

def _pHT_grad_trisH(solutearrays, tempK, pres, lnk_HSO4, lnk_trisH, lnk_MgOH,
        lnk_H2O):
    return derivative(lambda lnk_trisH: _solve_pHT(solutearrays, tempK, pres,
        lnk_HSO4, lnk_trisH, lnk_MgOH, lnk_H2O), lnk_trisH, dx=1e-6)

def _pHT_grad_MgOH(solutearrays, tempK, pres, lnk_HSO4, lnk_trisH, lnk_MgOH,
        lnk_H2O):
    return derivative(lambda lnk_MgOH: _solve_pHT(solutearrays, tempK, pres,
        lnk_HSO4, lnk_trisH, lnk_MgOH, lnk_H2O), lnk_MgOH, dx=1e-6)

def _pHT_grad_H2O(solutearrays, tempK, pres, lnk_HSO4, lnk_trisH, lnk_MgOH,
        lnk_H2O):
    return derivative(lambda lnk_H2O: _solve_pHT(solutearrays, tempK, pres,
        lnk_HSO4, lnk_trisH, lnk_MgOH, lnk_H2O), lnk_H2O, dx=1e-6)

def pHgrads(solutearrays, tempK, pres):
    print('Calculating pH gradients...')
    lnks = _getlnks(tempK)
    gradargs = (solutearrays, tempK, pres, *lnks)
    pHgrad_HSO4 = _pHT_grad_HSO4(*gradargs)
    print('{} = dpH/dK(HSO4)'.format(pHgrad_HSO4))
    pHgrad_trisH = _pHT_grad_trisH(*gradargs)
    print('{} = dpH/dK(trisH)'.format(pHgrad_trisH))
    pHgrad_MgOH = _pHT_grad_MgOH(*gradargs)
    print('{} = dpH/dK(MgOH)'.format(pHgrad_MgOH))
    pHgrad_H2O = _pHT_grad_H2O(*gradargs)
    print('{} = dpH/dK(H2O)'.format(pHgrad_H2O))
    return {
        'HSO4': pHgrad_HSO4,
        'trisH': pHgrad_trisH,
        'MgOH': pHgrad_MgOH,
        'H2O': pHgrad_H2O,
    }
