from setuptools import setup, find_packages

setup(
    name="memoripy",
    version="0.1.2",
    author="Khazar Ayaz",
    author_email="khazar.ayaz@personnoai.com",
    description="Memoripy provides context-aware memory management with support for OpenAI and Ollama APIs, offering structured short-term and long-term memory storage for interactive applications.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/caspianmoon/memoripy",
    packages=find_packages(),
    install_requires=[
        "aiohappyeyeballs",
        "aiohttp",
        "aiosignal",
        "annotated-types",
        "anyio",
        "attrs",
        "certifi",
        "charset-normalizer",
        "distro",
        # "faiss-cpu",
        "frozenlist",
        "h11",
        "httpcore",
        "httpx",
        "idna",
        "jiter",
        "joblib",
        "jsonpatch",
        "jsonpointer",
        "langchain",
        "langchain-core",
        "langchain-ollama",
        "langchain-openai",
        "langchain-text-splitters",
        "langsmith",
        "multidict",
        "networkx",
        "numpy",
        "ollama",
        "openai",
        "orjson",
        "packaging",
        "propcache",
        "pydantic",
        "pydantic_core",
        "PyYAML",
        "regex",
        "requests",
        "requests-toolbelt",
        "scikit-learn",
        "scipy",
        "setuptools",
        "sniffio",
        "SQLAlchemy",
        "tenacity",
        "threadpoolctl",
        "tiktoken",
        "tqdm",
        "typing_extensions",
        "urllib3",
        "yarl"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
