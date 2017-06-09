#from __future__ import absolute_import
#from __future__ import division
from __future__ import print_function
#
try:
    from setuptools import setup, Extension
except:
    from distutils.core import setup, Extension
#
from Cython.Build import cythonize
import numpy as np

ext = [# Pauli
       Extension("qmeq_elph.approach.c_pauli",
                 ["qmeq_elph/approach/c_pauli.pyx"]),
       # Lindblad
       Extension("qmeq_elph.approach.c_lindblad",
                 ["qmeq_elph/approach/c_lindblad.pyx"]),
       # Redfield
       Extension("qmeq_elph.approach.c_redfield",
                 ["qmeq_elph/approach/c_redfield.pyx"]),
       # 1vN
       Extension("qmeq_elph.approach.c_neumann1",
                 ["qmeq_elph/approach/c_neumann1.pyx"]),
       # Special functions
       Extension("qmeq_elph.specfuncc",
                 ["qmeq_elph/specfuncc.pyx"])]

cext = cythonize(ext)

setup(name='qmeq_elph',
      version='1.0',
      description='Package for transport calculations in quantum dots \
                   using approximate quantum master equations',
      #url='http://github.com/gedaskir/qmeq',
      author='Gediminas Kirsanskas',
      author_email='gediminas.kirsanskas@teorfys.lu.se',
      license='BSD 2-Clause',
      packages=['qmeq_elph', 'qmeq_elph/approach'],
      package_data={'qmeq_elph': ['*.pyx', '*.c', '*.pyd', '*.o', '*.so']},
      zip_safe=False,
      install_requires=['numpy', 'scipy', 'qmeq'],
      include_dirs = [np.get_include()],
      ext_modules = cext)
