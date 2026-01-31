from setuptools import find_packages,setup
from typing import List

a='-e .'
def get_requirements(file_path:str)->List[str]:
    requirement=[]
    with open(file_path) as obj:
        requirement=obj.readlines()
        requirement=[req.replace("\n","") for req in requirement]

        if a in requirement:
            requirement.remove(a)
    return requirement

setup(
    name="mlproject",
    version="0.0.1",
    author="jagadeesh",
    author_email="jagadeeshreddykonda55@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt')
)