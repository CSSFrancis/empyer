from setuptools import setup
from os import path
from io import open

here = path.abspath(path.dirname(__file__))


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='empyer',
      version='0.202hsReg',
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
                'empyer.simulate',
                'empyer.tests'],
      install_requires=['hyperspy >=1.3',
                        'numpy',
                        'matplotlib',
                        'scipy'],
      include_package_data=True,
      package_data={  # Optional
          'hspy_ext': ['hyperspy_extension.yaml'],
      },
      entry_points={'hyperspy.extensions': 'hspy_ext = hspy_ext'},
      zip_safe=False)
