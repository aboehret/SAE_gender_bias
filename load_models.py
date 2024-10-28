import torch as t
from nnsight import LanguageModel
from feature_circuits.dictionary_learning import AutoEncoder

class ModelLoader:
    """
    A class for loading a language model, its tokenizer, and relevant autoencoder dictionaries for each submodule.

    Attributes:
        model_name (str): The name of the language model to load.
        dictionary_path (str): Path where the dictionaries are stored.
        device (torch.device): Device to load the model ('cuda' or 'cpu').
        d_model (int): Dimension of the model layers.
        d_sae (int): Dimension of the autoencoder.
        dict_id (int): Dictionary ID used for loading specific dictionaries.
        model: The loaded language model.
        tokenizer: The tokenizer associated with the language model.
        dictionaries (dict): A dictionary containing the autoencoder for each submodule.
        submodules (list): A list of submodules in the language model (e.g., attention, MLP, residuals).
        submodule_names (dict): A dictionary mapping each submodule to its name.
    """
    def __init__(self, model_name, dictionary_path, device=None, d_model=512, d_sae=32768, dict_id=10):
        """
        Initialize the ModelLoader with model parameters.
        
        Args:
            model_name (str): The name of the language model to load.
            dictionary_path (str): Path where the dictionaries are stored.
            device (torch.device, optional): Device to load the model ('cuda' or 'cpu').
                                             Defaults to 'cuda' if available, otherwise 'cpu'.
            d_model (int, optional): Dimension of the model layers. Defaults to 512.
            d_sae (int, optional): Dimension of the autoencoder. Defaults to 32768.
            dict_id (int, optional): Dictionary ID used for loading specific dictionaries. Defaults to 10.
        """
        self.model_name = model_name
        self.dictionary_path = dictionary_path
        self.device = device if device else t.device("cuda" if t.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.d_sae = d_sae
        self.dict_id = dict_id
        
        self.model = None
        self.tokenizer = None
        self.dictionaries = {}
        self.submodules = []
        self.submodule_names = {}

    def load_model(self):
        """
        Load the language model and its tokenizer, and initialize submodules and dictionaries.
        
        Returns:
            tuple: A tuple containing:
                - model: The loaded language model.
                - tokenizer: The tokenizer associated with the model.
                - dictionaries (dict): A dictionary containing autoencoders for each submodule.
                - submodules (list): A list of the model's submodules.
                - submodule_names (dict): A mapping of submodules to their names.
        """
        self.model = LanguageModel(
            self.model_name,
            device_map=self.device,
            dispatch=True,
        )
        self.tokenizer = self.model.tokenizer  # Assuming tokenizer is part of the model
        self._load_submodules_and_dictionaries()
        return self.model, self.tokenizer, self.dictionaries, self.submodules, self.submodule_names

    def _load_submodules_and_dictionaries(self):
        """
        Load submodules (e.g., attention, MLP, residual layers) and their corresponding dictionaries.
        
        This method determines the submodules to load based on the specified model and initializes
        an autoencoder for each relevant submodule.
        
        Raises:
            ValueError: If the specified model name is not supported.
        """
        if self.model_name == "EleutherAI/pythia-70m-deduped":
            attns = [layer.attention for layer in self.model.gpt_neox.layers]
            mlps = [layer.mlp for layer in self.model.gpt_neox.layers]
            resids = [layer for layer in self.model.gpt_neox.layers]
            self.submodules = attns + mlps + resids

            for i in range(len(self.model.gpt_neox.layers)):
                # Load Attention dictionary
                self._load_autoencoder(attns[i], f'attn_{i}', f'attn_out_layer{i}')

                # Load MLP dictionary
                self._load_autoencoder(mlps[i], f'mlp_{i}', f'mlp_out_layer{i}')

                # Load Residual dictionary
                self._load_autoencoder(resids[i], f'resid_{i}', f'resid_out_layer{i}')
        elif self.model_name == "openai-community/gpt2":
            resids = [layer for layer in self.model.transformer.h]
            self.submodules = resids
            for i in range(len(self.model.transformer.h)):
                self._load_autoencoder(resids[i], f'resid_{i}', f'resid_out_layer{i}')
        else:
            raise ValueError('Model not supported')
        

    def _load_autoencoder(self, submodule, name, layer_name):
        """
        Helper method to load an autoencoder for a specific submodule.
        
        Args:
            submodule: The submodule (e.g., attention, MLP, residual) for which the autoencoder is being loaded.
            name (str): The name of the submodule.
            layer_name (str): The identifier for the layer used to locate the autoencoder file.
        
        Raises:
            FileNotFoundError: If the autoencoder file does not exist at the specified path.
        """
        ae = AutoEncoder(self.d_model, self.d_sae).to(self.device)
        if self.model_name == "EleutherAI/pythia-70m-deduped":
            dict_name = 'pythia-70m-deduped'
        ae_path = f'{self.dictionary_path}/dictionaries/{dict_name}/{layer_name}/{self.dict_id}_{self.d_sae}/ae.pt'
        ae.load_state_dict(t.load(ae_path))
        self.dictionaries[submodule] = ae
        self.submodule_names[submodule] = name
