import torch as t
import pandas as pd
from typing import List, Dict
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE 

class FeatureSteeringModule:
    """
    A class for generating text with and without feature steering in a language model.
    This module uses steering vectors to modify the model's activations during text generation.

    Attributes:
        device (torch.device): The device on which the model is loaded ('cuda' or 'cpu').
        model (HookedTransformer): The loaded language model that supports hook manipulation.
    """
    def __init__(self, model_name, device: None):
        """
        Initialize the FeatureSteeringModule with a language model.

        Args:
            model_name (str): The name of the language model to load.
            device (torch.device, optional): The device on which to load the model ('cuda' or 'cpu'). 
                                             Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        self.device = device if device is not None else t.device("cuda" if t.cuda.is_available() else "cpu")
        self.model = HookedTransformer.from_pretrained(model_name).to(device)

    def run_generate(
        self,
        example_prompt: str,
        hook_point: str,
        steering_vector: t.Tensor,
        coeff: float,
        sampling_kwargs: Dict,
        feature_id: int,
        detailed_output: bool = False,
    ):
        """
            Generate text with and without steering using a specified steering vector and coefficient.

            Args:
                example_prompt (str): The input prompt for text generation.
                hook_point (str): The point in the model to apply the steering.
                steering_vector (torch.Tensor): The vector used to steer the generation.
                coeff (float): The coefficient that controls the influence of the steering.
                sampling_kwargs (Dict): Additional arguments for the generation process (e.g., temperature, top_k).
                feature_id (int): The feature ID being used for steering.
                detailed_output (bool, optional): If True, prints detailed generation results. If False, returns a dictionary. 
                                                Defaults to False.

            Returns:
                dict: A dictionary containing the generation results with and without steering if `detailed_output` is False. 
                    The dictionary has the following keys:
                    - "feature_id": The ID of the feature used for steering.
                    - "coefficient": The coefficient applied to the steering vector.
                    - "with_steering": A list of generated texts with steering applied.
                    - "without_steering": A list of generated texts without steering.
            """
        result = {
            "feature_id": feature_id,
            "coefficient": coeff,
            "with_steering": [],
            "without_steering": []
        }

        self.model.reset_hooks()

        # Generation with steering
        editing_hooks = [
            (
                hook_point,
                lambda resid_post, hook: self.steering_hook(
                    resid_post, hook, steering_vector, coeff
                ),
            )
        ]
        res_with_steering = self.hooked_generate([example_prompt] * 3, editing_hooks, **sampling_kwargs)
        res_str_with_steering = self.model.to_string(res_with_steering[:, 1:])

        if detailed_output:
            print("Generation with steering:")
            print(("\n\n" + "-" * 80 + "\n\n").join(res_str_with_steering))
        else:
            result["with_steering"] = [self.model.tokenizer.decode(token) for token in res_with_steering[:, -1]]

        # Generation without steering
        res_without_steering = self.hooked_generate([example_prompt] * 3, [], **sampling_kwargs)
        res_str_without_steering = self.model.to_string(res_without_steering[:, 1:])

        if detailed_output:
            print("\nGeneration without steering:")
            print(("\n\n" + "-" * 80 + "\n\n").join(res_str_without_steering))
        else:
            result["without_steering"] = [self.model.tokenizer.decode(token) for token in res_without_steering[:, -1]]

        # Only return results if not printing detailed output
        if not detailed_output:
            return result

    def hooked_generate(
        self,
        prompt_batch: List[str],
        fwd_hooks: List,
        seed: int = None,
        max_new_tokens: int = 50,
        **kwargs,
    ) -> t.Tensor:
        """
        Helper function to generate text using a batch of prompts with optional forward hooks.

        Args:
            prompt_batch (List[str]): A list of input prompts for batch generation.
            fwd_hooks (List): A list of hooks to modify the model's forward pass.
            seed (int, optional): A seed for reproducibility of the generation. Defaults to None.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 50.
            **kwargs: Additional keyword arguments for the generation process.

        Returns:
            torch.Tensor: A tensor containing the generated tokens for each prompt in the batch.
        """
        if seed is not None:
            t.manual_seed(seed)

        with self.model.hooks(fwd_hooks=fwd_hooks):
            tokenized = self.model.to_tokens(prompt_batch)
            result = self.model.generate(
                stop_at_eos=False,
                input=tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                **kwargs,
            )
        return result

    @staticmethod
    def steering_hook(
        resid_post: t.Tensor,
        hook,
        steering_vector: t.Tensor,
        coeff: float,
    ):
        """
        Hook function to apply a steering effect to the model's residual stream based on a steering vector.

        Args:
            resid_post (torch.Tensor): The residual stream tensor at the hook point.
            hook (HookPoint): The hook point object, used to access the model's internals.
            steering_vector (torch.Tensor): The vector used to steer the residual stream.
            coeff (float): The coefficient that scales the steering vector's influence.
        """
        if resid_post.shape[1] > 1:
            resid_post[:, :-1, :] += coeff * steering_vector

    def run_with_layer_and_feature(
        self,
        prompt: str,
        layer: str,
        feature_id: int,
        coeff: float = 30,
        sampling_kwargs: Dict = None,
        table: bool = True,
    ):
        """
        Generate text using the specified layer and feature ID for steering.

        Args:
            prompt (str): The input prompt for text generation.
            layer (str): The layer in the model to apply steering (e.g., 'attn_2', 'mlp_3').
            feature_id (int): The feature ID to steer with.
            coeff (float, optional): The coefficient that scales the steering vector's influence. Defaults to 30.
            sampling_kwargs (Dict, optional): Additional arguments for the generation process. Defaults to None.
            table (bool, optional): If True, only returns results without detailed printing. Defaults to True.

        Returns:
            dict: A dictionary containing the generation results, similar to `run_generate`.
        """
        # Load the SAE model for the specified layer and feature
        sae = self._load_sae_model(layer)

        # Get the steering vector based on the feature ID
        steering_vector = sae.W_dec[feature_id]

        # Run the generation with steering
        return self.run_generate(
            example_prompt=prompt,
            hook_point=f"blocks.{layer}",
            steering_vector=steering_vector,
            coeff=coeff,
            sampling_kwargs=sampling_kwargs,
            feature_id=feature_id,
            detailed_output=not table
        )

    def _load_sae_model(self, layer: str):
        """
        Load the SAE model corresponding to a specified layer.

        Args:
            layer (str): The layer name (e.g., 'attn_2', 'resid_1').

        Returns:
            SAE: The loaded SAE model instance for the specified layer.

        Raises:
            ValueError: If the layer type is not recognized (must be 'attn', 'resid', or 'mlp').
        """
        if "attn" in layer:
            s_id = "pythia-70m-deduped-att-sm"
        elif "resid" in layer:
            s_id = "pythia-70m-deduped-res-sm"
        elif "mlp" in layer:
            s_id = "pythia-70m-deduped-mlp-sm"
        else:
            raise ValueError("Invalid layer type")

        sae, _, _ = SAE.from_pretrained(
            release=s_id,
            sae_id=f'blocks.{layer}',
            device='cpu',  # Replace 'cpu' with your device
        )
        return sae
