import os
from setuptools import setup


with open(os.path.join(os.path.dirname(__file__), 'LICENSE'), 'r') as f:
    license = f.read()


setup(name='hallgerd',
      version='0.1.1',
      description='Deep learning framework for OpenCL',
      author='Artem Razumov',
      author_email='airplaneless@yandex.ru',
      url='https://github.com/Airplaneless/Hallgerd',
      platforms=['any'],
      install_requires=['pyopencl', 'numpy>=1.13.0', 'tqdm', 'scikit-learn>=0.21.3', 'pybind11', 'mako'],
      packages=['hallgerd', 'gunnar', 'gunnar.kernels'],
      package_data={'gunnar.kernels': ['*.c']},
      data_files=[('gunnar', ['README.md']), ('', ['LICENSE', 'README.md'])],
      license=license
      )