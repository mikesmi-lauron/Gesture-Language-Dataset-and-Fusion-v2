import torch
import torch.nn as nn
from models.GLM import GLM
import numpy as np
import json
class MultiModalityMerger(nn.Module):
    """
    MultiModalityMerger processes multi-modal inputs (language & gesture)
    using a General Likelihood Merger (GLM). The model applies modality-specific
    weights, handles NaNs, and fuses features before producing output probabilities.
    """

    def __init__(self,method = "max",detection = "aod",config = None):
        """
        Initializes the MultiModalityMerger model.

        Attributes:
            glm (GLM): General Likelihood Merger for feature fusion.
            gesture_weight (nn.Parameter): Trainable weight for the gesture modality.
            language_weight (nn.Parameter): Trainable weight for the language modality.
        """
        super(MultiModalityMerger, self).__init__()
        self.method = method
        self.detection = detection
        """
        if self.detection == "aod":
            self.aod_ths = nn.Parameter(torch.randn(1))
        # Trainable modality-specific weights
        """
        if self.detection == "aod" or self.detection == "od":
            self.od_gesture_weight = nn.Parameter(torch.randn(1))
            self.od_language_weight = nn.Parameter(torch.randn(1))
        if self.detection == "aod" or self.detection == "ad":
            self.ad_gesture_weight = nn.Parameter(torch.randn(1))
            self.ad_language_weight = nn.Parameter(torch.randn(1))

        if method == "mix":
            if self.detection == "aod" or self.detection == "od":
                self.glm_od = GLM()
            if self.detection == "aod" or self.detection == "ad":
                self.glm_ad = GLM()

        elif method == "nn-1":
            hidden_size = 128
            num_columns = config["words_limit"] + 1  # Adjusted for an additional column
            num_rows = config["objects_limit"]
            output_size = num_rows  # Output size matches the number of objects
            
            if self.detection == "aod" or self.detection == "od":
                # Fully connected layers for processing combined language and gesture probabilities
                self.od_nn_merger = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(num_columns * num_rows, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 4, output_size)
                )
            if self.detection == "aod" or self.detection == "ad":
                actions_info = config["actions"]
                actions_tags = len(actions_info.keys())
                self.ad_nn_merger = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(actions_tags * 2, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 4, actions_tags)
                )
    def object_detection(self, language_probs, gesture_probs, scene_properties_matrix,mask):
        language_probs = torch.nan_to_num(language_probs, nan=0.0, posinf=0.0, neginf=0.0)
        gesture_probs = torch.nan_to_num(gesture_probs, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure gesture_probs is properly shaped for concatenation
        gesture_probs = gesture_probs.unsqueeze(-1)

        # Apply modality-specific weights
        language_probs *= self.od_language_weight
        gesture_probs *= self.od_gesture_weight

        # Transform language probabilities using scene properties matrix
        scene_properties_matrix_T = scene_properties_matrix.permute(0, 2, 1)
        language_probs = torch.matmul(scene_properties_matrix_T, language_probs)


        # Concatenate processed language and gesture probabilities
        combined_features = torch.cat([language_probs, gesture_probs], dim=-1)
        if self.method == "sum":
            output_logits = torch.sum(combined_features, dim=-1)
        elif self.method == "max":
            output_logits = torch.max(combined_features, dim=-1).values
        elif self.method == "mul":
            mask_mul = torch.any(combined_features != 0, dim=1, keepdim=True)
            combined_features_masked = combined_features.clone()
            mask_mul = mask_mul.expand(-1, combined_features_masked.shape[1], -1)
            combined_features_masked[~mask_mul] = 1
            output_logits = torch.prod(combined_features_masked, dim=-1)
        elif self.method == "mix":
            output_logits = self.glm_od(combined_features)
        elif self.method == "nn-1":
            output_logits = self.od_nn_merger(combined_features)
        else:
            raise NotImplementedError

        output_logits = output_logits.masked_fill(mask == 0, -1e9)
        output_probs = torch.softmax(output_logits, dim=1)

        return output_probs

    def action_detection_scene_context(self, sfm, afm):
        aom = torch.matmul(afm, sfm)
        aom = aom.clone()  # <-- Important! Clone it so you don't modify the computation graph
        aom[..., -1, :] = 0.99
        return aom

    def action_detection_apply_action_lh(self, alh, aom):
        max_values = torch.max(alh, dim=-1).values
        alh = alh.clone()  # <-- Clone before modifying
        alh[..., -1] = max_values
        alh = alh.unsqueeze(-1)
        alh = alh.expand(-1, -1, aom.shape[-1])
        aom_updated = aom * alh
        return aom_updated
    def action_detection_apply_object_lh(self,olh, aom):
        olh = olh.unsqueeze(-2)
        olh = olh.expand(-1,aom.shape[-2],-1)
        aom_updated = aom * olh
        return aom_updated

    def action_detection_lg_merger(self,l_alh,g_alh):
        l_alh_weigthed = l_alh * self.ad_language_weight
        g_alh_weighted = g_alh * self.ad_gesture_weight

        l_alh_weigthed = l_alh_weigthed.unsqueeze(-1)
        g_alh_weighted = g_alh_weighted.unsqueeze(-1)

        alh_combined = torch.cat([l_alh_weigthed, g_alh_weighted], dim=-1)
        if self.method == "sum":
            alh = torch.sum(alh_combined, dim=-1)
        elif self.method == "max":
            alh = torch.max(alh_combined, dim=-1).values
        elif self.method == "mul":
            mask = torch.any(alh_combined != 0, dim=1, keepdim=True)
            alh_masked = alh_combined.clone()
            mask = mask.expand(-1, alh_masked.shape[1], -1)
            alh_combined[~mask] = 1
            alh = torch.prod(alh_combined, dim=-1)
        elif self.method == "mix":
            alh = self.glm_ad(alh_combined)
        elif self.method == "nn-1":
            alh = self.ad_nn_merger(alh_combined)
        else:
            raise NotImplementedError
        alh = torch.softmax(alh, dim=-1)
        return alh
    def action_object_detection(self,olh,l_alh,g_alh,afm,sfm ):
        aom = self.action_detection_scene_context(sfm, afm)
        alh = self.action_detection_lg_merger(l_alh,g_alh)
        aom = self.action_detection_apply_action_lh(alh, aom)
        aom = self.action_detection_apply_object_lh(olh, aom)
        """
        aom[aom < self.aod_ths] = 0
        mask = (aom == 0).all(dim=(1, 2))
        zero_indices = mask.nonzero(as_tuple=True)[0]
        aom[zero_indices,-1,:] = olh[zero_indices]
        """
        return aom,alh
    def forward(self, inputs):
        if self.detection == "aod":
            olh = self.object_detection(*inputs[:4])
            aom,alh = self.action_object_detection(olh, *inputs[4:])
            aom_flatted = aom.view(aom.shape[0], -1)
            aom_flatted = aom_flatted / torch.max(aom_flatted)
            aom_flatted = torch.softmax(aom_flatted, dim=1)
            aom = aom_flatted.reshape(aom.shape)
            return aom,olh,alh
        elif self.detection == "ad":
            alh = self.action_detection_lg_merger(*inputs[4:6])
            return alh
        elif self.detection == "od":
            olh = self.object_detection(*inputs[:4])
            return olh

