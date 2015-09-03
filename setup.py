__author__ = 'WillCobb'
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='SimpleNeuralNetwork',

    version='0.4.0',

    description='A simple implemetation of a neural networking library',

    url='https://github.com/WilliamLCobb/SimpleNeuralNetwork',

    author='William Cobb',
    author_email='williamlewiscobb@gmail.com',

    license='MIT',

    classifiers=[

        'Development Status :: 3 - Alpha',


        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',


        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',

        "Operating System :: OS Independent"
    ],

    keywords='Simple Neural Network Machine Learning',


    packages=['SimpleNeuralNetwork'],

    #install_requires=['numpy'],
)