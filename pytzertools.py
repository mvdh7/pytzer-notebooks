from numpy import array

def solutes2arrays(solutes):
    fixions = array([sol for sol in solutes if not sol.startswith('t_')])
    fixmols = array([solutes[sol] for sol in solutes if not sol.startswith('t_')])
    eles = array([sol for sol in solutes if sol.startswith('t_')])
    tots = array([solutes[sol] for sol in solutes if sol.startswith('t_')])
    return fixions, fixmols, eles, tots

# tempK = np.array([tempK])
# pres = np.array([pres])
# allions = pz.properties.getallions(eles, fixions)
# allmxs = pz.matrix.assemble(allions, tempK, pres)
