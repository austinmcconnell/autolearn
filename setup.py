from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'Click >=6.7, <7.0',
    'matplotlib >=1.5, <2.0',
    'numpy >=1.11, <2.0',
    'pandas >=0.19, <2.0',
    'progressbar2 >=3.12, < 4.0',
    'scikit-learn >=0.18, <1.0',
    'seaborn >=0.7, <1.0'
]

test_requirements = [
    'coverage >=4.3, <5.0'
]

setup(
    name='autolearn',
    version='0.1.0',
    description="Automatic Machine Learning",
    long_description=readme,
    author="Austin McConnell",
    author_email='austin.s.mcconnell@gmail.com',
    url='https://github.com/austinmcconnell/autolearn',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'autolearn=autolearn.cli:main'
        ]
    },
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='autolearn',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
