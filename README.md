A Python code for efficiently evaluating the redshift-space power spectrum models presented in [arXiv:2105.12933](https://arxiv.org/abs/2105.12933) using the [FFTLog method](https://arxiv.org/abs/1708.08130).

As a by-product, this code can also return the one-loop density and velocity divergence power spectra (from [Eulerian perturbation theory](https://arxiv.org/abs/astro-ph/0112551)).

Functionality also exists for a user to supply their own one-loop integral to evaluate. To do this the kernel needs to be decomposed into FFTLog form (see appendix A in [arXiv:2105.12933](https://arxiv.org/abs/2105.12933) for an example on how to do this).
Then the kernel (and integral) is completely specified by a set of indices and coefficients, which is inputted into the function.

### Requirements
* numpy
* scipy
* pandas
* h5py