# Python Data Transforms

## Overview

This is a collection of functions that make it easy to do data transformations
using Jupyter notebooks in Python. Each of the different transformation functions
takes a Pandas dataset as input and returns one as output.

This also uses Plotly to output charts in the context of a Jupyter notebook.

This was originally created to help us migrate users off of Chartio and onto other
platforms Python-based data stacks. We thought this might be useful to the community.

## Usage

Take a look at transforms/ for a more complete listing of functions.

```python
import transforms

new_dataset = transforms.remove_columns(dataset, ["Column 1", "Column 2"])
```

## Contributing

This library is largely in maintenance mode. That said, if you would like to contribute,
please email us: team@hotswap.app. We'd love to hear from you!