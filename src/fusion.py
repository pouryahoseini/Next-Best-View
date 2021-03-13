import numpy as np


def fuse(input_1, input_2, fusion_type, universal_mass_1=None, universal_mass_2=None):
    """
    Fuse two class probability vectors.
    :param input_1: First input class probability vector (1D array)
    :param input_2: Second input class probability vector (1D array)
    :param fusion_type: Fusion type (str)
    :param universal_mass_1: First Universal object mass, in the case of Dempster-Shafer fusion
    :param universal_mass_2: Second Universal object mass, in the case of Dempster-Shafer fusion
    :return: Fused class probability vector (1D array)
    """

    # Check availability of Universal mass if the fusion type is Dempster-Shafer
    if fusion_type == 'DST' and (universal_mass_1 is None or universal_mass_2 is None):
        raise Exception("Universal mass not provided for computing the fused probability vector.")

    # Call th proper fusion function
    if fusion_type == 'Naive-Bayes':
        fused_probabilities = naive_bayes_fusion(input_1, input_2)
    elif fusion_type == 'Averaging':
        fused_probabilities = averaging_fusion(input_1, input_2)
    elif fusion_type == 'Dempster-Shafer':
        fused_probabilities = dempster_shafer_fusion(input_1, input_2, universal_mass_1, universal_mass_2)
    else:
        raise Exception("Unknown decision fusion type " + str(fusion_type) + ".")

    return fused_probabilities


def averaging_fusion(input_1, input_2):
    """
    Fuse two class probability vectors using averaging fusion.
    :param input_1: First input class probability vector (1D array)
    :param input_2: Second input class probability vector (1D array)
    :return: Fused class probability vector (1D array)
    """

    return (input_1 + input_2) / 2.0


def naive_bayes_fusion(input_1, input_2):
    """
    Fuse two class probability vectors using naive Bayes fusion (multiplication).
    :param input_1: First input class probability vector (1D array)
    :param input_2: Second input class probability vector (1D array)
    :return: Fused class probability vector (1D array)
    """

    # Multiply the two inputs
    multiplied_probabilities = input_1 * input_2

    # Scale the sum to 1
    fused_probabilities = multiplied_probabilities / np.sum(multiplied_probabilities)

    return fused_probabilities


def dempster_shafer_fusion(input_1, input_2, universal_mass_1, universal_mass_2):
    """
    Fuse two class mass vectors using Dempster-Shafer fusion to obtain a probability vector.
    :param input_1: First input class mass vector (1D array)
    :param input_2: Second input class mass vector (1D array)
    :param universal_mass_1: First Universal object mass, in the case of Dempster-Shafer fusion (int)
    :param universal_mass_2: Second Universal object mass, in the case of Dempster-Shafer fusion (int)
    :return: Fused class probability vector (1D array)
    """

    # Perform normalized Dempster's rule of combination
    unnormalized_fused_mass = input_1 * input_2 + universal_mass_1 * input_2 + universal_mass_2 * input_1
    normalization_factor = np.sum(unnormalized_fused_mass, dtype=np.float) + universal_mass_1 * universal_mass_2
    normalized_fused_mass = unnormalized_fused_mass / normalization_factor
    normalized_fused_universal_mass = (universal_mass_1 * universal_mass_2) / normalization_factor

    # Compute pignistic probabilities
    fused_probabilities = normalized_fused_mass + (normalized_fused_universal_mass / normalized_fused_mass.size)

    return fused_probabilities
