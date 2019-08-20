from distutils.core import setup

setup(name='hallgerd',
      version='0.1.0',
      packages=['hallgerd', 'gunnar', 'gunnar.kernels'],
      package_data={'gunnar.kernels': ['*.c']},
      data_files=[('hallgerd', ['README.md']), ('gunnar', ['README.md'])],
      )