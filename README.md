# Welcome to BlazingSQL Notebooks!

BlazingSQL Notebooks is a fully managed, high-performance JupyterLab environment. 

No setup required. You just login and start writing code, immediately.

Every environment has:
* An attached GPU
* Pre-Installed GPU Data Science Packages ([BlazingSQL](https://github.com/BlazingDB/blazingsql), [RAPIDS](https://github.com/rapidsai), [Dask](https://github.com/dask), and many more)

Below, we have listed a series of notebooks that demonstrate the utility of an open-source GPU powered Jupyter environment. Through the pre-installed packages you can execute familiar Python code on high-performance GPUs.

| Notebook | Description 
|----------------|----------------|
| [Welcome Notebook](welcome.ipynb) | An introduction to BlazingSQL Notebooks and the GPU Data Science Ecosystem.
| [The DataFrame](intro_notebooks/the_dataframe.ipynb) | Learn how to use BlazingSQL and cuDF to create GPU DataFrames with SQL and Pandas-like APIs.
| [Data Visualization](intro_notebooks/data_visualization.ipynb) | Plug in your favorite Python visualization packages, or use GPU accelerated visualization tools to render millions of rows in a flash.
| [Machine Learning](intro_notebooks/cuml.ipynb) | Learn about cuML, mirrored after the Scikit-Learn API, it offers GPU accelerated machine learning on GPU DataFrames.
| [Graph Analytics](intro_notebooks/cugraph.ipynb) | Run graph analytics on GPU DataFrames with cuGraph, which aims to provide a NetworkX-like API on GPU DataFrames.
| [Signal Analytics](intro_notebooks/cusignal.ipynb) | cuSignal is a direct port of     Scipy Signal built to leverage GPU compute resources through cuPy and Numba.
| [Cyber Security](intro_notebooks/clx.ipynb) | CLX ("clicks") provides a collection of RAPIDS examples for security analysts to quickly apply GPU acceleration to real-world cybersecurity use cases.
