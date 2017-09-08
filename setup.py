from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    README = readme_file.read()

with open('requirements.txt') as file:
    REQUIREMENTS = file.read()

setup(
    name='autolearn',
    version='0.1.0',
    description="Automatic Machine Learning",
    long_description=README,
    author="Austin McConnell",
    author_email='austin.s.mcconnell@gmail.com',
    url='https://github.com/austinmcconnell/autolearn',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'autolearn=autolearn.cli:main'
        ]
    },
    install_requires=REQUIREMENTS,
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
    test_suite='tests')
