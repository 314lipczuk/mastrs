import marimo

__generated_with = "0.21.1"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np


@app.function
def exported_func(x):
    return np.abs(x)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ok, the above is an exported function; We can make it work;
    """)
    return


if __name__ == "__main__":
    app.run()
