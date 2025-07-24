import json
import numpy as np
from data_creator.DataViz import DataViz

class Command:

    def __init__(self, target_objects = None, command_structure = None, type_id = None ,scene = None,cmd_scene_id = None):
        self.target_objs = target_objects
        self.structure = command_structure
        self.type_id = type_id
        self.dataset_cmd_id = scene.scene_id + "-" + str(cmd_scene_id) + "-" + str(type_id)
        self.meaning = None
        self.l_obj_props = []
        self.g_obj_probs = None
        self.g_pointing_coords = None

        self.target_actions = None
        self.action_id = None
        self.l_act_desc = None
        self.g_act_desc = None

    def update_action(self,action,config):
        #works only for single target obj
        self.action_id = action["id"]
        target_obj_features = self.target_objs[0]["features"]
        condition_meet = True
        for a_i in action["target_actions"]:
            keys = config["actions"][a_i].keys()

            for k in keys:
                if config["actions"][a_i][k] != target_obj_features[k]:
                    condition_meet = False
                    break
        if condition_meet:
            self.target_actions = action["target_actions"]
        else:
            self.target_actions = ["not_applicable"]
        self.l_act_desc = [action["language"]]
        self.meaning = [action["target_actions"][0]] + self.l_words
        self.g_act_desc = [action["gesture"]]
        self.dataset_cmd_id = self.dataset_cmd_id + f"_{self.action_id}"

    def update_l_obj_props(self):
        # udpate l_obj_props based on the structure
        target_obj = self.target_objs[0]
        for part in self.structure[1:]:
            self.l_obj_props.append(target_obj["properties"][part])

    def update_l_words(self):
        self.l_words =  self.l_obj_props

    def update_gpointing_coord(self,new_gpointing_coords = None ):
        if new_gpointing_coords is None:
            raise Exception("New g_pointing coord must be provided. See if you really want to do that")
        self.g_pointing_coords = new_gpointing_coords

    def update_g_obj_probs(self,new_g_obj_probs = None):
        if new_g_obj_probs is None:
            raise Exception("New g_obj_probs must be provided. See if you really want to do that")
        self.g_obj_probs = new_g_obj_probs



        all_probs = np.array([elem for elem in self.v_obj_probs if elem]).sum(axis=0)
        self.v_obj_total_probs = all_probs / np.sum(all_probs)
        self.v_obj_total_probs = self.v_obj_total_probs.tolist()

    def get_obj_scene_idx(self, scene):

        self.target_objs_scene_idx = []

        if self.target_objs is None:
            return

        for index, obj in enumerate(scene.objects):
            if obj == self.target_objs[0]:
                self.target_objs_scene_idx.append(index)

    def get_scene_id(self):
        if self.dataset_cmd_id is not None:
            ret= self.dataset_cmd_id.split("-")[0] +"-"+ self.dataset_cmd_id.split("-")[1]
            return ret

    def plot_command(self,scene,config,path2plot = None,object_index= None,ax = None):
        if scene.scene_id != self.get_scene_id():
            raise Exception("Scene id does not match command id")

        gesture_coord = self.gpointing_coord
        hand_coord = config["config"]["hand_position"]
        probs = [p for p in self.g_obj_probs if p != None][0]
        mask = [prob >= max(probs) for prob in probs]
        ax = DataViz.plot_3d_scene_with_views(scene, gesture_coord, hand_coord,
                                              path2plot=path2plot,
                                              sphere_color_mask = mask,ax = ax)
        if ax != None:
            return ax

    def from_json(self, command):
        self.dataset_cmd_id = command.get("id")
        self.type_id = command.get("type_id")
        self.action_id = command.get("action_id")
        self.set_scene_id(command.get("scene_id"))  # Assuming you have a setter method
        self.target_objs = [{"id": obj_id} for obj_id in command.get("target_objs_id", [])]
        self.target_actions = command.get("target_actions", [])
        self.meaning = command.get("meaning", "").split()
        self.l_obj_props = command.get("language_words_op", [[]])[0]
        self.l_act_desc = command.get("language_words_ad", [""])[0]
        self.g_act_desc = command.get("gesture_words_ad", [""])[0]
        self.g_obj_probs = command.get("g_pointing_obj_probs", [])


    def to_json(self):
        data = {
            "id": self.dataset_cmd_id,
            "type_id": self.type_id,
            "action_id": self.action_id,
            "scene_id": self.get_scene_id(),
            "target_objs_id": [obj["id"] for obj in self.target_objs],
            "target_actions": self.target_actions,
            "meaning": " ".join(self.meaning),
            "language_words_op" : [[w for w in self.l_obj_props if w != "None"]],
            "language_words_ad" : [self.l_act_desc],
            "gesture_words_ad" : [self.g_act_desc],
            "g_pointing_obj_probs" : self.g_obj_probs
        }
        return data

