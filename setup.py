from setuptools import setup, find_packages

import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likelyfail.'.format(sys.version_info.major))
          


setup(name='robustsp',
          version='0.1',
          description='library for robust signal processing',
          url='https://github.com/Mufabo/robustsp',
          author='M. Fatih Bostanci',
          author_email='fatih.bostanci@hotmail.de',
          license='MIT',
          packages= find_packages(),#['robustsp'],
          package_data = {'robustsp': ['data/*.mat']},
          install_requires=[
            'numpy',
            'matplotlib',
            'scipy',
            'setuptools',
            'statsmodels'
            ],
          zip_safe=False)