from setuptools import setup

setup(name='empyer',
      version='0.1',
      discription='Electron Microscopy tools for analyzing 4 and 5 dimensional STEM datasets',
      url='https://github.com/CSSFrancis/empyer',
      author='CSSFrancis',
      author_email='csfrancis@wisc.edu',
      liscense='UWMadison',
      packages=['empyer'],
      install_requires=['hyperspy','numpy', 'matplotlib', 'h5py'],
      zip_safe=False)
