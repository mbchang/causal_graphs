# causal_graphs

The generative model is

<img src="https://render.githubusercontent.com/render/math?math=x_1 \sim \mathcal{N}(0, I)">
<img src="https://render.githubusercontent.com/render/math?math=y \sim \mathcal{N}(f(x_1), I)">
<img src="https://render.githubusercontent.com/render/math?math=x_2 \sim \mathcal{N}(g(y), \sigma)">

where `f` and `g` are linear functions.

The learning problem is to predict `y` from `[x1, x2]`. If we use a linear model <img src="https://render.githubusercontent.com/render/math?math=y = w_1 \cdot x_1 + w_2 \cdot x_2 %2B b"> then the invariant solution should learn to set <img src="https://render.githubusercontent.com/render/math?math=w_2"> to 0.

We assume that the dimensionality of `x1` is 1.
To see how ERM with a linear model does as we vary the dimensionality of `x2` from 1 to `max_dim` and as we vary <img src="https://render.githubusercontent.com/render/math?math=\sigma"> from 1 to `max_std`, run: 

```
python spurious_correlations.py --max_dim <max_dim> --max_std <max_std>
```
