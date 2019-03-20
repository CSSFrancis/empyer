from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='empyer',
      version='0.1',
      description='Electron Microscopy tools for analyzing 4 and 5 dimensional STEM datasets',
      long_description=readme(),
      keywords='STEM Electron Microscopy Glass',
      url='https://github.com/CSSFrancis/empyer',
      author='CSSFrancis',
      author_email='csfrancis@wisc.edu',
      liscense='UWMadison',
      packages=['empyer'],
      install_requires=['hyperspy','numpy', 'matplotlib', 'h5py','scipy'],
      include_package_data=True,
      zip_safe=False)
