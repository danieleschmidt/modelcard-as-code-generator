"""Setup script for model card generator."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="modelcard-as-code-generator",
    version="1.0.0",
    description="Automated generation of Model Cards as executable, versioned artifacts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Terragon Labs",
    author_email="team@terragon-labs.com",
    url="https://github.com/terragon-labs/modelcard-as-code-generator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "click>=8.0.0",
        "pydantic>=1.8.0",
        "jinja2>=3.0.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "all": [
            "wandb>=0.15.0",
            "mlflow>=2.0.0",
            "dvc>=3.0.0",
            "huggingface-hub>=0.16.0",
            "pandas>=1.5.0",
            "numpy>=1.21.0",
        ],
        "wandb": ["wandb>=0.15.0"],
        "mlflow": ["mlflow>=2.0.0"],
        "dvc": ["dvc>=3.0.0"],
        "huggingface": ["huggingface-hub>=0.16.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "cli": ["click>=8.0.0"],
    },
    entry_points={
        "console_scripts": [
            "mcg=modelcard_generator.cli.main:main",
            "modelcard-generator=modelcard_generator.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Documentation",
        "Topic :: Software Development :: Quality Assurance",
    ],
    keywords=[
        "machine-learning", "model-cards", "ai-governance", "compliance",
        "documentation", "regulatory", "bias-detection", "model-validation"
    ],
    project_urls={
        "Bug Reports": "https://github.com/terragon-labs/modelcard-as-code-generator/issues",
        "Source": "https://github.com/terragon-labs/modelcard-as-code-generator",
        "Documentation": "https://terragon-labs.github.io/modelcard-as-code-generator/",
    },
    include_package_data=True,
    package_data={
        "modelcard_generator": [
            "templates/*.j2",
            "schemas/*.json",
            "compliance/*.yaml",
        ],
    },
    zip_safe=False,
)