# Welcome to BlazingSQL Notebooks!

BlazingSQL Notebooks is a fully managed, high-performance Jupyter notebook environment. 

No setup required. You just login and start writing code, immediately.

Every environment has:
* An attached GPU
* Pre-Installed GPU Data Science Packages ([BlazingSQL](https://blazingsql.com), [RAPIDS](https://rapids.ai), [Dask](https://dask.org), and many more)

Below, we have listed a series of notebooks that demonstrate the utility of a GPU powered Jupyter environment. Through the pre-installed packages you can execute familiar Python code on high-performance GPUs.

| Notebook | Description 
|----------------|----------------|
| [Welcome Notebook](welcome.ipynb) | An introduction to BlazingSQL Notebooks and the GPU Data Science Ecosystem.
| [The DataFrame](intro_notebooks/bsql_cudf.ipynb) | Learn how to use BlazingSQL and cuDF to create GPU DataFrames with SQL and Pandas-like APIs.
| [Data Visualization](intro_notebooks/bsql_cudf.ipynb) | Plug in your favorite Python visualization packages, or use GPU accelerated to render millions of rows in a flash.
| [Machine Learning](intro_notebooks/cuml.ipynb) | Learn about cuML, mirrored after the SK-Learn API, it offers GPU accelerated machine learning on GPU DataFrames.
| [cuML](intro_notebooks/cuml.ipynb) | Query 65M rows of network security data (netflow) with BlazingSQL and then pass to Graphistry to visualize and interact with the data |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/graphistry_netflow_demo.ipynb)|
| Taxi | Train a linear regression model with cuML on 55 million rows of public NYC Taxi Data loaded with BlazingSQL |coming soon|
| BlazingSQL vs. Apache Spark | Analyze 20 million rows of net flow data. Compare BlazingSQL and Apache Spark timings for the same workload |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/vs_pyspark_netflow.ipynb)|
| Federated Query | In a single query, join an Apache Parquet file, a CSV file, and a GPU DataFrame (GDF) in GPU memory. |[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/federated_query_demo.ipynb)|
[![Google Colab Badge](https://img.shields.io/badge/BSQL%20Notebooks-Launch%20on%20BlazingSQL%20Notebooks-green?style=flat-square&logo=appveyor&color=58585A)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/blazingsql_demo.ipynb)|
[![Google Colab Badge](https://img.shields.io/badge/BSQL%20Notebooks-Launch%20on%20BlazingSQL%20Notebooks-green?style=flat-square&logo=data:https://blazingsql.com/src/assets/blazingsql-icon.ico&color=58585A)](https://colab.research.google.com/github/BlazingDB/bsql-demos/blob/master/blazingsql_demo.ipynb)|

https://blazingsql.com/src/assets/blazingsql-icon.ico

