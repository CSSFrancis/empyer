from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='empyer',
      version='0.231',
      description='Electron Microscopy tools for analyzing 4 and 5 dimensional STEM datasets',
      long_description=readme(),
      keywords='STEM Electron Microscopy Glass',
      url='https://github.com/CSSFrancis/empyer',
      author='CSSFrancis',
      author_email='csfrancis@wisc.edu',
      liscense='MIT',
      packages=['empyer',
                'empyer.misc',
                'empyer.signals',
                'empyer.simulate'],
      install_requires=['hyperspy >=1.5',
                        'numpy>=1.10,!=1.70.0',
                        'matplotlib',
                        'scipy'],
      #  include_package_data=True, (this appearently breaks everything when you try to install the package :<)
      package_data={
          'empyer': ["hyperspy_extension.yaml"],
      },
      #  entry point is hyperspy.extensions, registering empyer as a
      entry_points={'hyperspy.extensions': ['empyer = empyer']},
      zip_safe=False)
