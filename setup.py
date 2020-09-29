from distutils.core import setup
setup(
    name='recoloradv',
    packages=[
        'recoloradv',
        'recoloradv.mister_ed',
        'recoloradv.mister_ed.utils',
    ],
    version='0.1',
    license='MIT',
    description='Attacks from the NeurIPS 2019 paper "Functional Adversarial Attacks"',
    author='Cassidy Laidlaw',
    author_email='claidlaw@umd.edu',
    url='https://github.com/cassidylaidlaw/ReColorAdv',
    download_url='https://github.com/cassidylaidlaw/ReColorAdv/archive/TODO.tar.gz',
    keywords=['adversarial examples', 'machine learning'],
    install_requires=[
        'torch>=1.4',
        'torchvision>=0.5',
        'matplotlib>=2.0.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)