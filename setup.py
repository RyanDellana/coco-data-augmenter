from setuptools import setup
from setuptools import find_packages

setup(name='cocoaugmenter',
      version='0.1.0',
      description='COCO data augmentation and sampling utilities',
      url='https://github.com/RyanDellana/coco-data-augmenter',
      author='Ryan Dellana',
      author_email='ryan.dellana@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=True)
