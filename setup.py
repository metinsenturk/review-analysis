from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
   name='review_analysis',
   version='0.0.9',
   description='Sentiment and topic modeling implementation on reviews.',
   long_description=readme(),
   url='https://github.com/metinsenturk/review-analysis',
   license='MIT',
   author='Metin Senturk',
   author_email='metinsenturk@me.com',
   packages=['review_analysis'],  #same as name
   dependency_links=['https://github.com/mimno/Mallet/archive/v2.0.8RC3.zip'],
   install_requires=[]
)