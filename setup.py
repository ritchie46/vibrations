from setuptools import setup

setup(
    name="vibrations",
    version="0.1a",
    author="Ritchie Vink",
    license='MIT License',
    url="https://ritchievink.com",
    download_url="https://github.com/ritchie46/vibrations",
    packages=["vibrations", "vibrations.ode"],
    install_requires=[
        "numpy>=1.11.1",
        "scipy>=0.18.1"
    ],
    description="Helper functions when processing vibration data"
)