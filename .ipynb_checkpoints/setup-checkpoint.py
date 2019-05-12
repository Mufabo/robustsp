from setuptools import setup, find_packages

setup(name='robustsp',
          version='0.1',
          description='library for robust signal processing',
          url='https://github.com/Mufabo/robustsp',
          author='M. Fatih Bostanci',
          author_email='fatih.bostanci@hotmail.de',
          license='MIT',
          packages= find_packages(),#['robustsp'],
          zip_safe=False)