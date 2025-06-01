from setuptools import setup, find_packages

setup(
    name="influencer",
    version="0.1.0",
    packages=find_packages() + ['diffusers_helper'],
    package_data={
        'diffusers_helper': ['**/*.py'],
    },
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "diffusers>=0.19.0",
        "TTS>=0.13.0",
        "moviepy>=1.0.3",
        "Pillow>=9.0.0",
        "safetensors>=0.3.1",
        "sentencepiece>=0.1.99",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.39.0",
    ],
    entry_points={
        "console_scripts": [
            "influencer=influencer.cli:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="AI Instagram Content Generator",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/influencer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 