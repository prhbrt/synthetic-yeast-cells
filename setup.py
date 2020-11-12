from setuptools import setup

setup(
    name='syntheticyeastcells',
    version='1.0',
    description='',
    url='https://github.com/prhbrt/synthetic-yeast-cells',
    author='Herbert Kruitbosch',
    author_email='H.T.Kruitbosch@rug.nl',
    packages=['syntheticyeastcells'],
    zip_safe=True,
    install_requires=[
        'opencv-python>=4.4.0.46',
        'opencv-contrib-python>=4.4.0.46',
        'tqdm>=4.51.0',
        'imgaug>=0.4.0',
        'pandas>=1.1.4',
    ],

)
