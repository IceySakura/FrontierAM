from pathlib import Path
import subprocess,sysconfig,numpy as np
root=Path(__file__).resolve().parents[1];out=root/'submissions/agent_sol_single'/('spd_core'+sysconfig.get_config_var('EXT_SUFFIX'))
headers=root/'.runtime/python-dev/usr/include'
includes=['-I'+sysconfig.get_path('include'),'-I'+np.get_include()]
if headers.exists():includes+=['-I'+str(headers/'python3.12'),'-I'+str(headers)]
subprocess.run(['c++','-O3','-march=native','-std=c++17','-fPIC','-fopenmp-simd','-shared',*includes,str(out.parent/'core.cpp'),'-o',str(out)],check=True)
print('frozen agent extension compiled for evaluator ABI')
