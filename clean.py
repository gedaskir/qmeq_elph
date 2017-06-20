import os
#import sys
import subprocess
import shutil
from glob import glob

dirs = ['./.cache',
        './build',
        './dist',
        './docs/build',
        './qmeq_elph.egg-info',
        './qmeq_elph/__pycache__',
        './qmeq_elph/build',
        './qmeq_elph/approach/__pycache__',
        './qmeq_elph/tests/__pycache__',
        './qmeq_elph/tests_/__pycache__']

for dr in dirs:
    try: shutil.rmtree(dr)
    except: pass

files = ( glob('./qmeq/*.pyd') + glob('./qmeq_elph/approach/*.pyd')
         +glob('./qmeq/*.pyc') + glob('./qmeq_elph/approach/*.pyc')
         +glob('./qmeq/*.c') + glob('./qmeq_elph/approach/*.c')
         +glob('./qmeq/*.html') + glob('./qmeq_elph/approach/*.html') )

for f in files:
    os.remove(f)
