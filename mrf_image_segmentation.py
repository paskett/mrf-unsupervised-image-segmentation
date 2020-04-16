"""Image Segmentation

The algorithm consists of four steps:
(i) re-segment the image,
(ii) sample noise model parameters,
(iii) sample MRF model parameters,
(iv) sample the number of classes.

We will implement each of those sections below.

If you need to access the paper, go here:
  https://static.aminer.org/pdf/PDF/000/180/822/unsupervised_image_segmentation_using_markov_random_field_models.pdf
"""

import numpy as np


def segment(img, numiter=100, alpha=(1.1, 10.0)):
    """Runs the entire segmentation algorithm.

    Inputs:

      img (ndarray) : a grayscale image to be segmented.

      numiter (int) : the number of iterations to compute.

      alpha (tuple (int)) : two constants used to calculate
              geometric annealing temperature.

    Outputs:

      segmentation (ndarray) : an array of the same shape as
            img where each entry is an integer indicating the
            class to which that pixel should belong.
    """

    # Initialize parameters.
    params = (None, None, None)

    # Segment the image.
    for t in range(numiter):
        # Obtain the temperature.
        T_t = (1 + alpha[0])**(alpha[1]*(1 - t/numiter))

        # Re-segment.
        segmentation = re_segment(img, t, alpha, params)

        # Sample noise parameters.
        noise_params = sample_noise() # returns a dict

        # Sample MRF parameters.
        MRF_params = sample_MRF() # returns a dict

        # Sample # of classes.
        num_classes = sample_num_classes() # returns an int

        # The parameters consist of noise model params,
        #  MRF model params, and the number of classes.
        params = (noise_parames, MRF_params, num_classes)

    return segmentation


def re_segment(img, t, alpha, params):
    """
    Returns the segmentation array by computing the probability
    of each pixel being in each class and returning the argmax
    """
    noise_params, MRF_params, num_classes = params

    # Obtain the temperature.
    T_t = (1 + alpha[0])**(alpha[1]*(1 - t/numiter))

    probs = np.zeros((num_classes, img.shape[0], img.shape[1]))

    for c in range(num_classes):
        # Get the parameters for this class.
        mu = noise_params['mu'][c]
        sig = noise_params['sig'][c]
        beta0 = MRF_params['beta0'][c]
        beta1 = MRF_params['beta1'][c]

        # Compute the probabilities for this class
        probs[idx] = 1/np.sqrt(2*np.pi*T_t*sig**2)*np.exp(
            -1/T_t*(0.5*((img - mu)/sig)**2
             + (beta0 + beta1*V(c, eta))) # FIXME: what's eta?
        ) # FIXME: define the function V()


    return np.argmax(probs, axis=0)

def noise_likelihood(img_c, mu, sig, T, n_c):
    return 1/(sig*(2*np.pi*sig**2*T)**n_c)*np.exp(-1/(2*T)*[((y-mu)/sig)**2 for y in img_c ])

def sample_noise(img, mu_0, sig_0, Y, T, n_c):
    """Samples the noise model parameters.
    The output is a dictionary with these keys:
    {
     'mu': an array with the mean for each class
     'sig': an array with the st. dev. for each class
    }"""
    means = []
    sigs = []
    for c in num_classes:
        img_c = np.ravel(img[np.where(Y==c)])
        mu = acceptance(img_c, mu_0, sig_0, T, n_c)
        means.append(mu)

        sig = acceptance(img_c, mu, sig_0, T, n_c)
        sigs.append(sig)

    return mus, sigs


def sample_MRF():
    """Samples the MRF model parameters.
    The output is a dictionary with these keys:
    {
     'beta0': an array with beta0 for each class
     'beta1': an array with beta1 for each class
    }
    """
    pass


def sample_num_classes():
    """Return an integer with the number of classes."""
    pass

def acceptance(old_sample, variance, likelihood_function):
    new_sample = np.random.normal(0, variance)

    old_likelihood = likelihood_function(old_sample)
    new_likelihood = likelihood_function(new_sample)

    ratio = new_likelihood/old_likelihood

    if ratio > 1:
        # always accept
        output = new_sample
    else:
        # accept w/ probability ratio
        draw = np.random.random()
        if draw < ratio:
            # accept
            output = new_sample
        else:
            # reject
            output = old_sample

    return output
