"""
Activation functions for neural networks.
"""
import numpy as np
#pylint: disable=no-name-in-module
from scipy.special import erf

def softmax(x, axis=0):
    
    kw = dict(axis=axis, keepdims=True)
    # make every value 0 or below, as exp(0) won't overflow
    try:
        xrel = x - x.max(**kw)
       
    except RuntimeWarning:
        
        xrel = x - x.max(**kw)
    
    # if you wanted better handling of small exponents, you could do something like this
    # to try and make the values as large as possible without overflowing, The 0.9
    # is a fudge factor to try and ignore rounding errors
    #
    #     xrel + = np.log(np.finfo(float).max / x.shape[axis]) * 0.9
    #print(x[0][0])
    exp_xrel = np.exp(xrel)
    return exp_xrel / exp_xrel.sum(**kw)


def softmax_prime(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def gelu(x):
    """
    Gaussian Error Linear Unit activation function.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def gelu_prime(x, approximate=False):
        r"""
        Evaluate the first derivative of the GELU function on the elements
        of input `x`.
        .. math::
            \frac{\partial \text{GELU}}{\partial x_i}  =
                \frac{1}{2} + \frac{1}{2}\left(\text{erf}(\frac{x}{\sqrt{2}}) +
                    \frac{x + \text{erf}'(\frac{x}{\sqrt{2}})}{\sqrt{2}}\right)
        where :math:`\text{erf}'(x) = \frac{2}{\sqrt{\pi}} \cdot \exp\{-x^2\}`.
        """
        pi, exp, sqrt, tanh = np.pi, np.exp, np.sqrt, np.tanh

        s = x / sqrt(2)
        erf_prime = lambda x: (2 / sqrt(pi)) * exp(-(x ** 2))  # noqa: E731

        if approximate:
            approx = tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3))
            dx = 0.5 + 0.5 * approx + ((0.5 * x * erf_prime(s)) / sqrt(2))
        else:
            dx = 0.5 + 0.5 * erf(s) + ((0.5 * x * erf_prime(s)) / sqrt(2))
        return dx
