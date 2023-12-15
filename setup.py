from setuptools import setup, find_packages

setup(
    name='ButaChanRL',
    version='0.0.1',
    packages=find_packages(),

    author='ThawTar',
    author_email='mr.thaw.tar1990@gmail.com',
    license='MIT',

    install_requires=[
        'gymnasium==0.28.1',
        'numpy>=1.25.2',
        'pandas==2.0.3',
        'matplotlib==3.7.2'
    ],

    #package_data={
    #    'ButaChanRL': ['datasets/data/*']
    #}
)