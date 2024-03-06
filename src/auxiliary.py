import numpy as np

def categorize(num_vector, bin=(18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100),
               int_bin=False):
    """
    Categorize each element of a numerical vector into predefined bins.

    Parameters:
    :param num_vector (array_like, float): A numpy.ndarray (or a structure that can be converted to it) containing
    numerical data to be categorized.
    :param bin (tuple of int/float, optional): A tuple defining the bin edges for categorization. Default is
    (18, 20, 25, 30, 35, ..., 90, 100).
    :param int_bin (bool, optional): If True, categories are returned as integers representing bin indices;
    if False, categories are returned as strings showing bin ranges. Default is False.

    Returns:
    :return: A numpy.ndarray containing the categorized data. The data type of the array is either 'int'
    (if int_bin is True) or 'str' (if int_bin is False).
    """
    bin = np.array(bin, dtype=np.float32)
    num_vector = np.array(num_vector)

    if not np.all(np.sort(bin) == bin):
        raise ValueError("Categories should be in increasing order")
    if len(bin) < 2:
        warnings.warn("Vector remains unchanged; Expected len(bin)>=2.")
        return num_vector

    # Check for out of range values
    if np.any(num_vector < bin[0]) or np.any(num_vector > bin[-1]):
        raise ValueError(f"Values should be within the range [{bin[0]}, {bin[-1]}]")

    # Assign bin
    category_labels = [f"[{int(cat1)},{int(cat2)})" for cat1, cat2 in zip(bin, bin[1:])]
    category_labels[-1] = f"[{int(bin[-2])},{int(bin[-1])}]"

    if int_bin:
        # skip the first left bin edge bin[0] since digitize assumes (-Inf, bin[0])
        # skip the right most bin[-1], to include values x=bin[-1] to the last interval [bin[-2], bin[-1]]
        categorized = np.digitize(num_vector, bin[1:-1], right=False)
    else:
        categorized = np.array([category_labels[i] for i in np.digitize(num_vector, bin[1:-1])])
    return np.array(categorized)
