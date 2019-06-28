from setuptools import find_packages, setup

setup(
    name='koalearn',
    version='0.0.1',
    description='My private package for learning',
    url='https://github.com/koaly/koalearn.git',
    author='Natthapong Somboonphattarakit',
    author_email='koaly@koaly.me',
    license="Wisesight",
    packages=find_packages(),
    zip_safe=False

    package_data={"koalearn": ["model/*", "vectorizer/*"]},
)