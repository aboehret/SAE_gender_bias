import torch as t
import itertools

class Metrics:
    """
    A collection of static methods for calculating metrics related to faithfulness, completeness,
    and logit differences in model outputs.
    """
    @staticmethod
    def faithfulness(fc, fempty, fm):
        """
        Calculates the faithfulness of an ablation, comparing the effect of specific features against
        the baseline and full model.

        Args:
            fc (float): The output value when specific features are ablated.
            fempty (float): The output value when no features are ablated.
            fm (float): The output value of the full model without ablation.
        
        Returns:
            float: A score indicating the faithfulness of the ablation.
        """
        return ((fc - fempty) / (fm - fempty))

    @staticmethod
    def completeness(fccomp, fempty, fm):
        """
        Calculates the completeness of an ablation, evaluating how completely the specific features
        account for the model's behavior.

        Args:
            fccomp (float): The output value when complementary features are ablated.
            fempty (float): The output value when no features are ablated.
            fm (float): The output value of the full model without ablation.
        
        Returns:
            float: A score indicating the completeness of the ablation.
        """
        return ((fccomp - fempty) / (fm - fempty))
    
    @staticmethod
    def logit_diff_metric(model, clean_answers, patch_answers):
        """
        Computes the average logit difference between clean and patched answers.

        Args:
            model: The language model being evaluated.
            clean_answers (Tensor): Tensor containing indices of clean answers.
            patch_answers (Tensor): Tensor containing indices of patched answers.
        
        Returns:
            Tensor: The mean difference in logits for the clean and patched answers.
        """
        return t.mean(
            model.embed_out.output[:,-1, patch_answers] - model.embed_out.output[:,-1, clean_answers],
            dim = -1
        )


class AblationManager:
    """
    A class for managing feature ablation experiments, allowing for the calculation
    of metrics to assess the causal impact of individual or sets of features.
    """
    def __init__(self, model, dictionaries, submodules):
        """
        Initializes the AblationManager with a model, dictionaries for feature encoding/decoding,
        and submodule references.

        Args:
            model: The language model used for ablation.
            dictionaries (dict): A dictionary of encoding/decoding dictionaries for each submodule.
            submodules (list): A list of submodule names involved in the ablation.
        """
        self.model = model
        self.dictionaries = dictionaries
        self.submodules = submodules

    def compute_ablation(self, clean_input_ids, patch_input_ids, metric_fn, ablation_idxs, keep_idxs=True, metric_kwargs={}):
        """
        Performs feature ablation on the model's submodules, using the specified metric function to evaluate impact.

        Args:
            clean_input_ids (Tensor): Tensor of input IDs for the clean condition.
            patch_input_ids (Tensor, optional): Tensor of input IDs for the patched condition.
            metric_fn (callable): A metric function used to evaluate the model after ablation.
            ablation_idxs (dict): A dictionary specifying indices to ablate for each submodule.
            keep_idxs (bool, optional): Whether to keep (True) or remove (False) the specified indices. Defaults to True.
            metric_kwargs (dict, optional): Additional keyword arguments for the metric function. Defaults to {}.

        Returns:
            Tensor: The values of the specified metric after ablation.
        """
        is_tuple = {}
        with self.model.trace("_"):
            for submodule in ablation_idxs:
                is_tuple[submodule] = type(submodule.output.shape) == tuple

        patch_states = {}
        if patch_input_ids is not None:
            with t.no_grad(), self.model.trace(patch_input_ids):
                for submodule in self.submodules:
                    dictionary = self.dictionaries[submodule]
                    x = submodule.output
                    if type(x.shape) == tuple:
                        x = x[0]
                    x_hat, f = dictionary(x, output_features=True)
                    patch_states[submodule] = f.save()

        with t.no_grad(), self.model.trace(clean_input_ids):
            for submodule in ablation_idxs:
                dictionary = self.dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                f = dictionary.encode(x)
                residual = x - dictionary(x)

                if keep_idxs:
                    mask = t.zeros_like(f, dtype=t.bool)
                    mask[..., ablation_idxs[submodule]] = True
                else:
                    mask = t.ones_like(f, dtype=t.bool)
                    mask[..., ablation_idxs[submodule]] = False
                if patch_input_ids is not None:
                    f[..., ~mask] = patch_states[submodule].value[..., ~mask]
                else:
                    f[..., ~mask] = t.zeros_like(f)[..., ~mask]

                x_hat = dictionary.decode(f)
                x_recon = residual + x_hat
                if is_tuple[submodule]:
                    submodule.output[0][:] = x_recon
                else:
                    submodule.output = x_recon

            metric_values = metric_fn(self.model, **metric_kwargs).save()
        return metric_values

    def run_ablation(self, clean_prefixes, patch_prefixes, clean_answers, patch_answers, ablation_idxs, keep_idxs=True):
        """
        Runs an ablation experiment for the specified indices and returns the average logit difference metric.

        Args:
            clean_prefixes (Tensor): Tensor of prefixes for the clean condition.
            patch_prefixes (Tensor): Tensor of prefixes for the patched condition.
            clean_answers (Tensor): Tensor of clean answers' indices.
            patch_answers (Tensor): Tensor of patched answers' indices.
            ablation_idxs (dict): A dictionary specifying indices to ablate for each submodule.
            keep_idxs (bool, optional): Whether to keep (True) or remove (False) the specified indices. Defaults to True.

        Returns:
            float: The mean value of the logit difference metric after ablation.
        """
        metric_values_clean = self.compute_ablation(
            clean_input_ids=clean_prefixes,
            patch_input_ids=patch_prefixes,
            metric_fn= Metrics.logit_diff_metric,
            ablation_idxs=ablation_idxs,
            keep_idxs=keep_idxs,
            metric_kwargs={'clean_answers': clean_answers, 'patch_answers': patch_answers},
        )
        return metric_values_clean.mean()

    def compute_ablation_onebyone(self, clean_prefixes, patch_prefixes, clean_answers, patch_answers, ablation_id):
        """
        Computes the ablation metrics for a single feature at a time.

        Args:
            clean_prefixes (Tensor): Tensor of prefixes for the clean condition.
            patch_prefixes (Tensor): Tensor of prefixes for the patched condition.
            clean_answers (Tensor): Tensor of clean answers' indices.
            patch_answers (Tensor): Tensor of patched answers' indices.
            ablation_id (dict): A dictionary specifying a single feature's index to ablate for each submodule.

        Returns:
            tuple: Contains values for fempty, fmodel, fc, faith, fcomp, comp, and bias.
        """
        dataset = dict(
            clean_prefixes=clean_prefixes,
            patch_prefixes=patch_prefixes,
            clean_answers=clean_answers,
            patch_answers=patch_answers,
        )

        ablation_idxs = {submodule: t.tensor([], dtype=t.int) for submodule in self.submodules}

        fempty = self.run_ablation(**dataset, ablation_idxs=ablation_idxs, keep_idxs=True)
        fmodel = self.run_ablation(**dataset, ablation_idxs=ablation_idxs, keep_idxs=False)

        ablation_idxs2 = {submodule: t.tensor([], dtype=t.int) for submodule in self.submodules}
        for k, v in ablation_id.items():
            ablation_idxs2[self.submodules[k]] = t.tensor([v], dtype=t.int)

        fc = self.run_ablation(**dataset, ablation_idxs=ablation_idxs2, keep_idxs=True)
        fcomp = self.run_ablation(**dataset, ablation_idxs=ablation_idxs2, keep_idxs=False)

        faith = Metrics.faithfulness(fc, fempty, fmodel)
        comp = Metrics.faithfulness(fcomp, fempty, fmodel)
        bias = (fmodel - fcomp) / fmodel

        return fempty, fmodel, fc, faith, fcomp, comp, bias
    
    def compute_ablation_layerwise(self, clean_prefixes, patch_prefixes, clean_answers, patch_answers, ablation_layer, D_SAE):
        """
        Performs a layerwise ablation, analyzing the impact of removing all features in a specified layer.

        Args:
            clean_prefixes (Tensor): Tensor of prefixes for the clean condition.
            patch_prefixes (Tensor): Tensor of prefixes for the patched condition.
            clean_answers (Tensor): Tensor of clean answers' indices.
            patch_answers (Tensor): Tensor of patched answers' indices.
            ablation_layer (int): The index of the layer to ablate.
            D_SAE (int): The dimensionality of the autoencoder output for the layer.
        
        Returns:
            tuple: Contains values for fempty, fmodel, fc, faith, fcomp, comp, and bias.
        """
        dataset = dict(
            clean_prefixes=clean_prefixes,
            patch_prefixes=patch_prefixes,
            clean_answers=clean_answers,
            patch_answers=patch_answers,
        )

        ablation_idxs = {submodule: t.tensor([], dtype=t.int) for submodule in self.submodules}

        # Compute baseline and skyline
        fempty = self.run_ablation(**dataset, ablation_idxs=ablation_idxs, keep_idxs=True)
        fmodel = self.run_ablation(**dataset, ablation_idxs=ablation_idxs, keep_idxs=False)

        # Create ablation mask for the specific layer
        ablation_idxs[self.submodules[ablation_layer]] = t.tensor([i for i in range(D_SAE)], dtype=t.int)

        fc = self.run_ablation(**dataset, ablation_idxs=ablation_idxs, keep_idxs=True)
        fcomp = self.run_ablation(**dataset, ablation_idxs=ablation_idxs, keep_idxs=False)

        faith = Metrics.faithfulness(fc, fempty, fmodel)
        comp = Metrics.completeness(fcomp, fempty, fmodel)
        bias = (fmodel - fcomp) / fmodel

        return fempty, fmodel, fc, faith, fcomp, comp, bias
    
    def compute_ablation_submodules(self, clean_prefixes, patch_prefixes, clean_answers, patch_answers, ablation_submodules, D_SAE):
        """
        Performs ablation across multiple submodules, analyzing the impact of feature removal.

        Args:
            clean_prefixes (Tensor): Tensor of prefixes for the clean condition.
            patch_prefixes (Tensor): Tensor of prefixes for the patched condition.
            clean_answers (Tensor): Tensor of clean answers' indices.
            patch_answers (Tensor): Tensor of patched answers' indices.
            ablation_submodules (list): A list of submodule indices to ablate.
            D_SAE (int): The dimensionality of the autoencoder output.
        
        Returns:
            tuple: Contains values for fempty, fmodel, fc, faith, fcomp, comp, and bias.
        """
        dataset = dict(
            clean_prefixes=clean_prefixes,
            patch_prefixes=patch_prefixes,
            clean_answers=clean_answers,
            patch_answers=patch_answers,
        )

        ablation_idxs = {submodule: t.tensor([], dtype=t.int) for submodule in self.submodules}

        # Compute baseline and skyline
        fempty = self.run_ablation(**dataset, ablation_idxs=ablation_idxs, keep_idxs=True)
        fmodel = self.run_ablation(**dataset, ablation_idxs=ablation_idxs, keep_idxs=False)

        # Create ablation mask for the specified submodules
        for submodule in ablation_submodules:
            ablation_idxs[self.submodules[submodule]] = t.tensor([i for i in range(D_SAE)], dtype=t.int)

        fc = self.run_ablation(**dataset, ablation_idxs=ablation_idxs, keep_idxs=True)
        fcomp = self.run_ablation(**dataset, ablation_idxs=ablation_idxs, keep_idxs=False)

        faith = Metrics.faithfulness(fc, fempty, fmodel)
        comp = Metrics.completeness(fcomp, fempty, fmodel)
        bias = (fmodel - fcomp) / fmodel

        return fempty, fmodel, fc, faith, fcomp, comp, bias

    def compute_ablation_featureset(self, clean_prefixes, patch_prefixes, clean_answers, patch_answers, ablation_ids):
        """
        Performs ablation on a specific set of features, analyzing the impact of their removal.

        Args:
            clean_prefixes (Tensor): Tensor of prefixes for the clean condition.
            patch_prefixes (Tensor): Tensor of prefixes for the patched condition.
            clean_answers (Tensor): Tensor of clean answers' indices.
            patch_answers (Tensor): Tensor of patched answers' indices.
            ablation_ids (list): A list of dictionaries specifying which features to ablate for each submodule.
        
        Returns:
            tuple: Contains values for fempty, fmodel, fc, faith, fcomp, comp, and bias.
        """
        dataset = dict(
            clean_prefixes=clean_prefixes,
            patch_prefixes=patch_prefixes,
            clean_answers=clean_answers,
            patch_answers=patch_answers,
        )

        ablation_idxs = {submodule: t.tensor([], dtype=t.int) for submodule in self.submodules}

        # Compute baseline and skyline
        fempty = self.run_ablation(**dataset, ablation_idxs=ablation_idxs, keep_idxs=True)
        fmodel = self.run_ablation(**dataset, ablation_idxs=ablation_idxs, keep_idxs=False)

        # Create ablation mask for the specified feature set
        ablation_idxs2 = {submodule: t.tensor([], dtype=t.int) for submodule in self.submodules}
        for ablation_id in ablation_ids:
            for k, v in ablation_id.items():
                ablation_idxs2[self.submodules[k]] = t.tensor([v], dtype=t.int)

        fc = self.run_ablation(**dataset, ablation_idxs=ablation_idxs2, keep_idxs=True)
        fcomp = self.run_ablation(**dataset, ablation_idxs=ablation_idxs2, keep_idxs=False)

        faith = Metrics.faithfulness(fc, fempty, fmodel)
        comp = Metrics.faithfulness(fcomp, fempty, fmodel)
        bias = (fmodel - fcomp) / fmodel

        return fempty, fmodel, fc, faith, fcomp, comp, bias

    def ablate_feature_combinations(self, biased_feat, clean_prefixes, patch_prefixes, clean_answers, patch_answers):
        """
        Tests combinations of biased features through ablation, assessing the impact on bias metrics.

        Args:
            biased_feat (list): A list of biased features to test in combination.
            clean_prefixes (Tensor): Tensor of prefixes for the clean condition.
            patch_prefixes (Tensor): Tensor of prefixes for the patched condition.
            clean_answers (Tensor): Tensor of clean answers' indices.
            patch_answers (Tensor): Tensor of patched answers' indices.
        
        Returns:
            tuple: Contains a list of bias scores for each feature combination and a list of scores including faith and comp.
        """
        dataset = dict(
            clean_prefixes=clean_prefixes,
            patch_prefixes=patch_prefixes,
            clean_answers=clean_answers,
            patch_answers=patch_answers,
        )

        bias_scores = []
        all_scores = []

        for r in range(1, len(biased_feat) + 1):
            for combination in itertools.combinations(biased_feat, r):
                ablation_ids = [{submodule: idx} for feature in combination for submodule, idx in feature.items()]

                fempty, fmodel, fc, faith, fcomp, comp, bias = self.compute_ablation_featureset(
                    **dataset, ablation_ids=ablation_ids
                )

                bias_scores.append({'combination': ablation_ids, 'bias_score': bias.item()})
                all_scores.append({
                    'combination': ablation_ids,
                    'bias_score': bias.item(),
                    'faith_score': faith.item(),
                    'comp_score': comp.item()
                })

        return bias_scores, all_scores
