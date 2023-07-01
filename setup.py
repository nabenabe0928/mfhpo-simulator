import setuptools


requirements = []
with open("requirements-for-setup.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())


setuptools.setup(
    name="mfhpo-simulator",
    version="1.2.1",
    author="nabenabe0928",
    author_email="shuhei.watanabe.utokyo@gmail.com",
    url="https://github.com/nabenabe0928/mfhpo-simulator",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    platforms=["Linux", "Darwin"],
    install_requires=requirements,
    include_package_data=True,
)
