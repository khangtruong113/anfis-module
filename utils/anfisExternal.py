from tensorflow import random_uniform, Variable, random_normal, reduce_mean, Tensor


# Initialize variables as Premise Parameters
def premise_parameters(para_shape,
                       min_mu=10.0, max_mu=15.0,
                       min_sig=5.0, max_sig=10.0):
    """
    :param para_shape:
    :param min_mu:
    :param max_mu:
    :param min_sig:
    :param max_sig:
    :return:
    """
    para_init = {
        'mu': Variable(random_uniform(para_shape, minval=min_mu, maxval=max_mu)),
        'sigma': Variable(random_uniform(para_shape, minval=min_sig, maxval=max_sig))
    }
    return para_init


# Initialize variables as Consequence Parameters
def consequence_parameters(para_shape
                           ):
    para_init = random_normal(para_shape)
    return Variable(para_init)


# Get neighbors
def get_neighbor(x: Tensor):
    delta = random_normal(shape=x.get_shape(), mean=0.0, stddev=0.001 * reduce_mean(x))
    x = x + delta
    return x
