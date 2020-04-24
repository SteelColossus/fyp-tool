# Software Performance Prediction Comparision (SPPC) Tool
This tool was developed for my Final Year Project. It allows for comparison of different machine learning methods for the domain of predicting software performance.

## Prerequisites
1. Clone the repository locally.
2. Install [Python 3.6](https://www.python.org/downloads/release/python-368/) and ensure that pip is installed as well.
3. Install [Tensorflow 1.10](https://www.tensorflow.org/install/pip). (You can optionally set up a virtual environment for this.)
4. Install [scikit-learn](https://scikit-learn.org/stable/install.html).
5. Install the other required libraries, [matplotlib](https://matplotlib.org/users/installing.html#installing-an-official-release), [tabulate](https://pypi.org/project/tabulate/) and [psutil](https://pypi.org/project/psutil/).

## How to run
This tool is run via Python at the command line, e.g.

```
python sppc_tool.py LLVM
```

Replace `LLVM` with whatever software system you want to get predictions for. (The program looks in the `/data` folder)

To get help with the command or to see optional command line arguments, run:

```
python sppc_tool.py --help
```
