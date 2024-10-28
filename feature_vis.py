import urllib.parse
import webbrowser
import json


def transform_list(data_dict, model_id="pythia-70m-deduped"):
    """
    Transforms a dictionary of features into a list format suitable for use with external tools.

    Args:
        data_dict (dict): A dictionary where keys are component indices (e.g., 'attn_0', 'resid_1') 
                          and values are dictionaries of feature indices for each component.
        model_id (str, optional): The identifier of the model for which features are extracted. 
                                  Defaults to "pythia-70m-deduped".

    Returns:
        list: A list of dictionaries, each containing:
            - "modelId": The model identifier.
            - "layer": A string representing the layer in a transformed format (e.g., '0-att-sm').
            - "index": The index of the feature as a string.
    """
    
    layer_mapping = {
        'attn_': 'att-sm',
        'resid_': 'res-sm',
        'mlp_': 'mlp-sm'
    }
    result = []
    
    for component_idx, feature_indices in data_dict.items():
        # Determine the type of layer based on the key
        layer = component_idx
        layer_prefix = layer[:-1]
        layer_suffix = layer[-1]
        layer_key = f"{layer_mapping.get(layer_prefix, layer_prefix)}"

        # Add the information for each index in this layer
        for index in feature_indices.keys():
            result.append({
                "modelId": model_id,
                "layer": f"{layer_suffix}-{layer_key}",
                "index": str(index)
            })

    return result


def get_neuronpedia_quicklist(features_list, LIST_NAME = "Gender features"):
    """
    Opens a web browser with a URL that creates a quick list of features on Neuronpedia.

    Args:
        features_list (list): A list of feature dictionaries containing information about the features.
        LIST_NAME (str, optional): The name of the list to be created on Neuronpedia. 
                                   Defaults to "Gender features".

    Returns:
        None: This function has no return value, but it opens a URL in the web browser to create a quick list.
    """
    LIST_FEATURES = features_list

    url = "https://neuronpedia.org/quick-list/"
    name = urllib.parse.quote(LIST_NAME)
    url = url + "?name=" + name
    url = url + "&features=" + urllib.parse.quote(json.dumps(LIST_FEATURES))

    print("Opening: " + url)
    webbrowser.open(url)