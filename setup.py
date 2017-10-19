
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

try:
    from setuptools_rust import Binding, RustExtension
    rust_available = True
except ImportError:
    rust_available = False

default_setup_kwargs = dict(
    name='gap-stat',
    version='1.1.0',
    author='Miles Granger',
    maintainer='Miles Granger',
    author_email='miles.granger@outlook.com',
    maintainer_email='miles.granger@outlook.com',
    keywords='kmeans unsupervised learning machine-learning clustering',
    description='Python implementation of the gap statistic.',
    long_description='Uses the gap statistic method by Tibshirani, Walther, Hastie to suggest n_clusters.',
    packages=['gap_statistic'],
    license='BSD',
    url='https://github.com/milesgranger/gap_statistic',
    zip_safe=True,
    install_requires=['numpy', 'pandas', 'scipy'],
    tests_require=['pytest', 'scikit-learn', 'pytest-runner'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta'
    ]
)

if not rust_available:
    setup(**default_setup_kwargs)
else:
    print('-' * 10 + 'Building rust extension' + '-' * 10)
    default_setup_kwargs.update({'zip_safe': False})
    setup(rust_extensions=[RustExtension('gap_statistic.rust_gap', 'Cargo.toml', binding=Binding.PyO3)],
          **default_setup_kwargs
          )

