from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='empyer',
      version='0.153',
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
                        'matplotlib'],
      include_package_data=True,
      zip_safe=False)
