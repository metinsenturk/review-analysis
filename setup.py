from setuptools import setup, find_packages, find_namespace_packages

def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
   name='review-analysis',
   version='0.0.9',
   description='Sentiment and topic modeling implementation on reviews.',
   long_description=readme(),
   url='https://github.com/metinsenturk/review-analysis',
   license='MIT',
   author='Metin Senturk',
   author_email='metinsenturk@me.com',
   packages=find_packages(),
   package_dir={'review-analysis':'src'},
   include_package_data = False,
   dependency_links=['https://github.com/mimno/Mallet/archive/v2.0.8RC3.zip'],
   install_requires=requirements()
)