import setuptools


requirements = []
with open("requirements-for-setup.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())

pkgs = [
    "benchmark_simulator",
    "benchmark_simulator/_simulator",
    "benchmark_simulator/utils",
    "benchmark_simulator/_trackers",
]

setuptools.setup(
    name="mfhpo-simulator",
    version="1.4.0",
    author="nabenabe0928",
    author_email="shuhei.watanabe.utokyo@gmail.com",
    url="https://github.com/nabenabe0928/mfhpo-simulator",
    packages=pkgs,
    python_requires=">=3.8",
    platforms=["Linux", "Darwin"],
    install_requires=requirements,
    include_package_data=True,
)
