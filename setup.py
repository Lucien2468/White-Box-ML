from setuptools import setup, find_packages

setup(
    name='white_box_ml',
    version='0.1.0',
    description='A transparent, interpretable machine learning library',
    author='Lucien Hu',
    author_email='lucienhu2222@gmail.com',
    url='https://github.com/Lucien2468/white-box-ml',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'reversegrad>=0.1.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
