import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="facecrop",
    version="0.0.1",
    author="Utt Assoratgoon",
    author_email="uttasso@gmail.com",
    description="Croping face package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UdbT/FaceCrop",
    packages=setuptools.find_packages(),
    install_requires=[
                        "cycler==0.10.0",
                        "dlib==19.19.0",
                        "joblib==1.2.0",
                        "kiwisolver==1.2.0",
                        "matplotlib==3.2.1",
                        "numpy==1.18.4",
                        "opencv-contrib-python==4.2.0.34",
                        "pyparsing==2.4.7",
                        "python-dateutil==2.8.1",
                        "scikit-learn==0.23.0",
                        "scipy==1.4.1",
                        "six==1.14.0",
                        "threadpoolctl==2.0.0"
                    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)