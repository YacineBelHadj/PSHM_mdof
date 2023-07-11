# Deep Learning Project Template

This template offers a lightweight yet functional project template for various deep learning projects.
The template assumes [PyTorch](https://pytorch.org/) as the deep learning framework.
However, one can easily transfer and utilize the template to any project implemented with other frameworks.

## Table of Contents

- [Getting Started](#getting-started)
- [Template Layout](#template-layout)
- [Resources](#resources)
- [Authors](#authors)
- [License](#license)

## Getting Started

You can fork this repo and use it as a template when creating a new repo on Github like this:
<p align="center">
    <img src="https://github.com/xduan7/dl-project-template/blob/master/docs/readme/create_a_new_repo.png" width="90%">
</p>
Or directly use the template from the forked template repo like this:
<p align="center">
    <img src="https://github.com/xduan7/dl-project-template/blob/master/docs/readme/use_template.png" width="90%">
</p>

Alternatively, you can simply download this repo in zipped format and get started:
<p align="center">
    <img src="https://github.com/xduan7/dl-project-template/blob/master/docs/readme/download.png" width="90%">
</p>

Next, you can install all the libraries.
Check out the [Extra Packages](#extra-packages) writeup for some awesome packages.

To export and install the dependencies, you can use the following command:
```bash
make freeze
make install
```

There are some other useful commands automated with the makefile. 
Please check out the usage with the following command:
```bash
make help
```

## Template Layout

```text
dl-project-template
.
├── configs             # configuration files (for conda, flake8, etc.)
├── data
│   ├── ...             # data reference files (index, readme, etc.)
│   ├── raw             # untreated data directly downloaded from source
│   ├── interim         # intermediate data processing results
│   └── processed       # processed data (features and targets) ready for learning
├── docs                # documentation files (*.txt, *.doc, *.jpeg, etc.)
├── exps                # experiments with configuration files
├── logs                # logs for deep learning experiments
├── models              # saved models with optimizer states 
├── notebooks           # Jupyter Notebooks (mostly for data processing and visualization)
├── src    
│   ├── data            # data processing classes, functions, and scripts
│   ├── eval            # evaluation classes and functions (metrics, visualization, etc.)
│   ├── modules         # activations, layers, modules, and networks (subclass of torch.nn.Module)
│   └── utils           # other useful functions and classes
├── tests               # unit tests module for ./src
├── makefile            # makefile for various commands (install, test, check, etc.) 
├── license.md
└── readme.md
```

## Resources

The [resources](./docs/readme/resources.md) file includes packages, datasets, readings, and other templates.

## Authors

* Xiaotian Duan (Email: xduan7 at gmail.com)

## License

This project is licensed under the MIT License. Check [LICENSE.md](LICENSE.md) for more details.

