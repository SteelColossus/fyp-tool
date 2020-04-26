# Software Performance Prediction Comparision (SPPC) Tool
This tool was developed for my Final Year Project. It allows for comparison of different machine learning methods for the domain of predicting software performance.

## Prerequisites
1. Clone the repository locally.
2. Install [Python 3.6](https://www.python.org/downloads/release/python-368/) and ensure that pip is installed as well.
3. Install [Tensorflow 1.10](https://www.tensorflow.org/install/pip). (You can optionally set up a virtual environment for this.)
4. Install [scikit-learn](https://scikit-learn.org/stable/install.html).
5. Install the other required libraries, [matplotlib](https://matplotlib.org/users/installing.html#installing-an-official-release), [tabulate](https://pypi.org/project/tabulate/) and [psutil](https://pypi.org/project/psutil/).
6. Optionally, clone the [DeepPerf](https://github.com/DeepPerf/DeepPerf) repository, create a `/extensions` folder in the tool's directory and copy the DeepPerf repository into it. This will allow using the DeepPerf tool as a model to train with.

## How to run
This tool is run via Python at the command line, e.g.

```
python sppc_tool.py LLVM
```

Replace `LLVM` with whatever software system you want to get predictions for. (The program looks in the `/data` folder)

For convenience, several aliases are provided for the existing files in this folder (these are case insensitive):
- `Apache_Storm`: the Apache Storm Word Count dataset (`SS-K1.csv`)
- `FPGA_Sort`: the FPGA Sort dataset (`SS-B2.csv`)
- `LLVM`: the LLVM dataset (`SS-L1.csv`)
- `SaC`: the Seismic Analysis Code dataset (`SS-O2.csv`)
- `Trimesh`: the Trimesh dataset (`SS-M2.csv`)
- `X264-DB`: the X264 dataset (`SS-N1.csv`)

To get help with the command or to see optional command line arguments, run:

```
python sppc_tool.py --help
```
