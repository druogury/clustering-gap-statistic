
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='gap-stat',
      version='1.2.0',
      author='Miles Granger',
      maintainer='Damien RUSSIER',
      author_email='miles.granger@outlook.com',
      maintainer_email='damien.russier@gmail.com',
      keywords='kmeans unsupervised learning machine-learning clustering',
      description='Python implementation of the gap statistic.',
      long_description='Uses the gap statistic method by Tibshirani, Walther, Hastie to suggest n_clusters.',
      packages=['gap_statistic'],
      license='BSD',
      url='https://github.com/druogury/clustering-gap-statistic',
      zip_safe=True,
      install_requires=['numpy', 'pandas', 'scipy', 'spherecluster'],
      setup_requires=['pytest-runner'],
      tests_require=['coverage', 'pytest', 'scikit-learn'],
      classifiers=[
            'Programming Language :: Python :: 3',
            'Development Status :: 4 - Beta'
      ]
      )
