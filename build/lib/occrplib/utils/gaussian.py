from typing import Dict, Iterable, Tuple
from scipy.stats import norm


def fit_gaussian_model(probabilities: Iterable[float]) -> Tuple[float, float]:
    """Fits a gaussian (normal) model with mean 1

    Args:
        probabilities (Iterable[float]): list of probabilities to be fitted

    Returns:
        Tuple[float, float]: mean and standard deviation of fitted normal distribution
    """
    prob_pos = [p for p in probabilities] + [2 - p for p in probabilities]
    fitted_mean, fitted_std = norm.fit(prob_pos)
    return fitted_mean, fitted_std


def get_gaussian_parameters_per_class(training_data_classes: Dict[str, Iterable[float]]) -> Dict[str, Tuple[float, float]]:
    """Obtain the mean and average for each class from a probabilitiers dictionary

    Args:
        training_data_classes (Dict[str, Iterable[float]]): dictionary containing
            probabilities for each class. Keys are classes (string labels)

    Returns:
        Dict[str, Tuple[float, float]]: dictionary with mean and standard deviation
            for each class
    """
    return {i: fit_gaussian_model(probs) for i, probs in training_data_classes.items()}


def get_classes_thresholds(training_data_classes: Dict[str, Iterable[float]], scale: float = 1) -> Dict[str, float]:
    """Obtain threshold for each class using gaussian fitting

    Args:
        training_data_classes (Dict[str, Iterable[float]]): dictionary containing
            probabilities for each class. Keys are classes (string labels)
        scale (float, optional): number of standard deviations to compute the thresholds.
            Higher values mean lower thresholds. Defaults to 1.

    Returns:
        Dict[str, float]: _description_
    """

    fitting = get_gaussian_parameters_per_class(training_data_classes)
    return {i: max(0.5, 1.0 - scale * std) for i, (mean, std) in fitting.items()}
