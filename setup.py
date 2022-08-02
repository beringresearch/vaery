from setuptools import setup, find_packages

setup(name='vaery',
      version='0.0.1',
      description='Synthetic data generator',
      url='http://github.com/beringresearch/vaery',
      author='Bering Limited',
      license='Apache 2.0',
      packages=find_packages(),
      ###python_requires='>=3.5',
      install_requires=['numpy', 
                        'scikit-learn>0.20.0',
                        'tensorflow',
                        'tensorflow-io', 
                        'ipywidgets'],
      zip_safe=False)



#      extras_require={
#          'tests': ['pytest'],
#          'visualization': ['matplotlib', 'seaborn'],
#          'cpu': ['tensorflow-cpu'],
#          'gpu': ['tensorflow']
#      },
