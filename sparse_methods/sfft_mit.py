MIT_SFFT_LIMITATION = (
    "MIT sFFT 1.0/2.0 source archive is a 1D research code with manually tuned "
    "per-(n,k) parameters; it does not provide a usable 2D matrix sparse-FFT API here."
)


def sfft_mit_topk_features(*args, **kwargs):
    raise NotImplementedError(MIT_SFFT_LIMITATION)
