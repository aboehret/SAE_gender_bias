import torch as t
import numpy as np

def get_all_features(nodes, submodule_names):
    """
    Extracts all non-zero features from a set of nodes.

    Args:
        nodes (dict): A dictionary where keys are component indices and values are activation objects.
        submodule_names (dict): A mapping from component indices to submodule names.
    
    Returns:
        dict: A dictionary with submodule names as keys and dictionaries of feature indices and their values as values.
    """
    feature_indices = {}
    for component_idx, effect in nodes.items(): 
        feature_indices[submodule_names[component_idx]] = {}
        for idx in (effect).nonzero():
            feature_indices[submodule_names[component_idx]][idx.item()] = effect[idx].item()
    return feature_indices

def get_thres_features(nodes, submodule_names, threshold= 0.1):
    """
    Extracts features whose absolute activation exceeds a specified threshold.

    Args:
        nodes (dict): A dictionary where keys are component indices and values are activation objects.
        submodule_names (dict): A mapping from component indices to submodule names.
        threshold (float, optional): The minimum absolute activation value to include a feature. Defaults to 0.1.
    
    Returns:
        dict: A dictionary with submodule names as keys and dictionaries of filtered feature indices and their values as values.
    """
    feature_indices = {}
    for component_idx, effect in nodes.items():
        feature_indices[submodule_names[component_idx]] = {}
        for idx in (t.abs(effect.act) > threshold).nonzero():
            feature_indices[submodule_names[component_idx]][idx.item()] = effect[idx].item()
    return feature_indices


def get_topk_features(nodes, submodule_names, top_n):
    """
    Retrieves the top-k features with the highest absolute activations across all submodules.

    Args:
        nodes (dict): A dictionary where keys are component indices and values are activation objects.
        submodule_names (dict): A mapping from component indices to submodule names.
        top_n (int): The number of top features to extract based on absolute activation.
    
    Returns:
        dict: A dictionary with submodule names as keys and dictionaries of the top-k feature indices and their values as values.
    """
    feature_dict = {submodule: {} for submodule in sorted(submodule_names.values())}
    all_features = []
    for component_idx, effect in nodes.items():
        for idx, act in enumerate(effect.act):
            all_features.append((component_idx, idx, act.item()))
    top_features = sorted(all_features, key=lambda x: abs(x[2]), reverse=True)[:top_n]
    for component_idx, feature_idx, effect_value in top_features:
        submodule_name = submodule_names[component_idx]
        if submodule_name in feature_dict:
            feature_dict[submodule_name][feature_idx] = effect_value

    return feature_dict


def get_top_features(nodes, top_n, submodule_names):
    """
    Retrieves the top-n features with the highest absolute activations within each submodule.

    Args:
        nodes (dict): A dictionary where keys are component indices and values are activation objects.
        top_n (int): The number of top features to extract per submodule.
        submodule_names (dict): A mapping from component indices to submodule names.
    
    Returns:
        dict: A dictionary with submodule names as keys and dictionaries of the top-n feature indices and their values as values.
    """
    feature_indices = {}
    for component_idx, effect in nodes.items():
        top_indices = t.abs(effect.act).argsort(descending=True)[:top_n]
        feature_indices[submodule_names[component_idx]] = {
            idx.item(): effect.act[idx].item() for idx in top_indices
        }
    return feature_indices


def get_stratified_features(nodes, submodule_names, num_features=30, seed=None):
    """
    Selects a stratified set of non-zero features from the provided nodes.
    
    Args:
        nodes (dict): A dictionary where keys are component indices and values are objects containing activations.
        submodule_names (dict): A mapping from component indices to submodule names.
        num_features (int): The number of random features to select.
        seed (int, optional): A seed for reproducibility. Defaults to None.
    
    Returns:
        dict: A dictionary with submodule names as keys and dictionaries of selected features as values.
    """
    if seed is not None:
        np.random.seed(seed)

    all_features = []
    for component_idx, effect in nodes.items():
        activations = np.array([act.item() for act in effect.act])
        non_zero_mask = activations != 0  
        if non_zero_mask.any():
            non_zero_indices = np.nonzero(non_zero_mask)[0]  
            non_zero_activations = activations[non_zero_mask] 
            all_features.extend(zip([component_idx] * len(non_zero_indices), non_zero_indices, non_zero_activations))
    
    if not all_features:
        return {}

    all_features = np.array(all_features, dtype=object)
    activations = all_features[:, 2].astype(float) 
    sorted_indices = np.argsort(activations) 
    sorted_features = all_features[sorted_indices]
    
    # Define stratification thresholds (33rd and 66th percentiles)
    num_samples = min(num_features, len(all_features))
    third = len(sorted_features) // 3
    
    low_features = sorted_features[:third]
    middle_features = sorted_features[third:2*third]
    high_features = sorted_features[2*third:]
    low_sample_size = num_samples // 3
    middle_sample_size = num_samples // 3
    high_sample_size = num_samples - (low_sample_size + middle_sample_size)  
    
    selected_low = low_features[np.random.choice(len(low_features), min(low_sample_size, len(low_features)), replace=False)]
    selected_middle = middle_features[np.random.choice(len(middle_features), min(middle_sample_size, len(middle_features)), replace=False)]
    selected_high = high_features[np.random.choice(len(high_features), min(high_sample_size, len(high_features)), replace=False)]
    selected_features = np.concatenate([selected_low, selected_middle, selected_high])
    feature_dict = {submodule: {} for submodule in sorted(submodule_names.values())}
    
    for component_idx, feature_idx, activation in selected_features:
        submodule_name = submodule_names[component_idx]
        feature_dict[submodule_name][int(feature_idx)] = float(activation)
    
    return feature_dict


def get_random_features(nodes, submodule_names, num_features=30, seed=None):
    """
    Selects a random set of non-zero features from the provided nodes.
    
    Args:
        nodes (dict): A dictionary where keys are component indices and values are objects containing activations.
        submodule_names (dict): A mapping from component indices to submodule names.
        num_features (int): The number of random features to select.
        seed (int, optional): A seed for reproducibility. Defaults to None.
    
    Returns:
        dict: A dictionary with submodule names as keys and dictionaries of selected features as values.
    """
    if seed is not None:
        np.random.seed(seed)
    
    all_features = []

    for component_idx, effect in nodes.items():
        activations = np.array([act.item() for act in effect.act])
        non_zero_mask = activations != 0 
        
        if non_zero_mask.any():
            non_zero_indices = np.nonzero(non_zero_mask)[0]  
            non_zero_activations = activations[non_zero_mask] 
            all_features.extend(zip([component_idx] * len(non_zero_indices), non_zero_indices, non_zero_activations))
    
    if not all_features:
        return {}
    
    all_features = np.array(all_features, dtype=object)
    num_samples = min(num_features, len(all_features))
    selected_indices = np.random.choice(len(all_features), num_samples, replace=False)
    selected_features = all_features[selected_indices]
    feature_dict = {submodule: {} for submodule in sorted(submodule_names.values())}
    
    for component_idx, feature_idx, activation in selected_features:
        submodule_name = submodule_names[component_idx]
        feature_dict[submodule_name][int(feature_idx)] = float(activation)
    
    return feature_dict


def get_diff(dict1, dict2):
    """
    Compares two dictionaries of features to find differences.

    Args:
        dict1 (dict): The first dictionary with submodule names as keys and feature dictionaries as values.
        dict2 (dict): The second dictionary with submodule names as keys and feature dictionaries as values.
    
    Returns:
        dict: A dictionary indicating features present only in `dict1` (value 1) or only in `dict2` (value 2).
    """
    differing_features = {}
    
    for submodule in dict1.keys():
        if submodule in dict2:
            features1 = set(dict1[submodule].keys())
            features2 = set(dict2[submodule].keys())
            diff1 = features1 - features2  
            diff2 = features2 - features1  
            differing_features[submodule] = {}
            for feature in diff1:
                differing_features[submodule][feature] = 1
            for feature in diff2:
                differing_features[submodule][feature] = 2
    
    return differing_features


def get_sim(dict1, dict2):
    """
    Compares two dictionaries of features to find similarities.

    Args:
        dict1 (dict): The first dictionary with submodule names as keys and feature dictionaries as values.
        dict2 (dict): The second dictionary with submodule names as keys and feature dictionaries as values.
    
    Returns:
        dict: A dictionary indicating common features with a value of 1.
    """
    common_features = {}
    
    for submodule in dict1.keys():
        if submodule in dict2:
            features1 = set(dict1[submodule].keys())
            features2 = set(dict2[submodule].keys())
            common = features1.intersection(features2)
            common_features[submodule] = {feature: 1 for feature in common}
    
    return common_features

