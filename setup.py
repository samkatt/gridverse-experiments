#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "gym_gridverse @ git+ssh://git@github.com/abaisero/gym-gridverse.git@fe6d5dfa4ec62e893d6335c62f4d2b8d2cec4159",
    "online_pomdp_planning @ git+ssh://git@github.com/samkatt/online-pomdp-planning.git@master",
    "pomdp_belief_tracking @ git+ssh://git@github.com/samkatt/pomdp-belief-tracking.git@main",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="sammie katt",
    author_email="sammie.katt@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A package that matches solution methods library with gym-gridverse",
    entry_points={
        "console_scripts": [
            "gridverse_experiments=gridverse_experiments.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="gridverse_experiments",
    name="gridverse_experiments",
    packages=find_packages(
        include=["gridverse_experiments", "gridverse_experiments.*"]
    ),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/samkatt/gridverse_experiments",
    version="0.1.0",
    zip_safe=False,
)
