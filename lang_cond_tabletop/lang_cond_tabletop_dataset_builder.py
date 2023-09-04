from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os
import json
from PIL import Image
import random


class LangCondTabletop(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        self.action_inst_to_verb = {
            'push': ['push', 'move'],
            'pick': ['pick', 'pick up', 'raise', 'hold'],
            'put_down': ['put down', 'place down']
        }
        self.noun_phrase = {
            0: {
                'name': ['red', 'maroon'],
                'object': ['object', 'cube', 'square'],
            },
            1: {
                'name': ['red', 'coke', 'cocacola'],
                'object': ['can', 'bottle'],
            },
            2: {
                'name': ['blue', 'pepsi'],
                'object': ['can', 'bottle'],
            },
            3: {
                'name': ['milk', 'white'],
                'object': ['carton', 'box'],
            },
            4: {
                'name': ['bread', 'yellow object', 'brown object'],
                'object': [''],
            },
            5: {
                'name': ['green', '', 'glass', 'green glass'],
                'object': ['bottle'],
            }
        }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of [6x robot joint angles, '
                                '1x gripper position].',
                        ),
                        'state_vel': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint velocity, consists of [6x robot joint angles, '
                                '1x gripper position].',
                        )
                    }),
                    'ground_truth_states': tfds.features.FeaturesDict({
                        'EE': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='xyzrpy',),

                        'cube': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='xyzrpy',),

                        'coke': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='xyzrpy',),

                        'pepsi': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='xyzrpy',),

                        'milk': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='xyzrpy',),

                        'bread': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='xyzrpy',),

                        'bottle': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='xyzrpy',),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'action_delta': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot delta action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                    'goal_object': tfds.features.Text(
                        doc='Object to be manipulated with.'
                    ),
                    'action_inst': tfds.features.Text(
                        doc='Action to be performed.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='real_ur5'),
        }


    def noun_phrase_template(self, target_id):

        id_name = np.random.randint(len(self.noun_phrase[target_id]['name']))
        id_object = np.random.randint(len(self.noun_phrase[target_id]['object']))
        name = self.noun_phrase[target_id]['name'][id_name]
        obj = self.noun_phrase[target_id]['object'][id_object]
        return (name + ' ' + obj).strip()

    def verb_phrase_template(self, action_inst):
        action_id = np.random.randint(len(self.action_inst_to_verb[action_inst]))
        verb = self.action_inst_to_verb[action_inst][action_id]
        return verb.strip()

    def sentence_template(self, target_obj, action_inst):
        self.target_name_to_idx = {
            'target2': 0,
            'coke': 1,
            'pepsi': 2,
            'milk': 3,
            'bread': 4,
            'bottle': 5,
        }
        sentence = ''
        verb = self.verb_phrase_template(action_inst)
        sentence = sentence + verb
        sentence = sentence + ' ' + self.noun_phrase_template(self.target_name_to_idx[target_obj])
        return sentence.strip()

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        name_conversion_dict = {
            'EE': 'EE',
            'target2': 'cube',
            'coke': 'coke',
            'pepsi': 'pepsi',
            'milk': 'milk',
            'bread': 'bread',
            'bottle': 'bottle',
        }

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
    
            states_json = os.path.join(episode_path, 'states.json')
            with open(states_json) as json_file:
                states_dict = json.load(json_file)
                json_file.close()
            img_paths = [os.path.join(episode_path, 'real_' + str(i) + '_processed.png') for i in range(len(states_dict) - 1)]
            
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            
            for step_idx in range(len(states_dict) - 1):
                lang_inst = self.sentence_template(states_dict[step_idx]['goal_object'], states_dict[step_idx]['action_inst'],)
                episode.append({
                    'observation': {
                        'image': np.array(Image.open(img_paths[step_idx]))[:,:,:3],
                        'state': states_dict[step_idx]['q'],
                        'state_vel': states_dict[step_idx]['dq'],
                    },
                    'ground_truth_states': {
                        'EE': np.array(states_dict[step_idx]['objects_to_track']['EE']['xyz'] + states_dict[step_idx]['objects_to_track']['EE']['rpy'], dtype=np.float32),
                        'cube': np.array(states_dict[step_idx]['objects_to_track']['target2']['xyz'] + states_dict[step_idx]['objects_to_track']['EE']['rpy'], dtype=np.float32),
                        'coke': np.array(states_dict[step_idx]['objects_to_track']['coke']['xyz'] + states_dict[step_idx]['objects_to_track']['EE']['rpy'], dtype=np.float32),
                        'pepsi': np.array(states_dict[step_idx]['objects_to_track']['pepsi']['xyz'] + states_dict[step_idx]['objects_to_track']['EE']['rpy'], dtype=np.float32),
                        'milk': np.array(states_dict[step_idx]['objects_to_track']['milk']['xyz'] + states_dict[step_idx]['objects_to_track']['EE']['rpy'], dtype=np.float32),
                        'bread': np.array(states_dict[step_idx]['objects_to_track']['bread']['xyz'] + states_dict[step_idx]['objects_to_track']['EE']['rpy'], dtype=np.float32),
                        'bottle': np.array(states_dict[step_idx]['objects_to_track']['bottle']['xyz'] + states_dict[step_idx]['objects_to_track']['EE']['rpy'], dtype=np.float32),
                    },
                    'action': np.array(states_dict[step_idx + 1]['objects_to_track']['EE']['xyz'] + states_dict[step_idx + 1]['objects_to_track']['EE']['rpy'] + [states_dict[step_idx + 1]['q'][-1]], dtype=np.float32),
                    'action_delta': np.array(states_dict[step_idx + 1]['objects_to_track']['EE']['xyz'] + states_dict[step_idx + 1]['objects_to_track']['EE']['rpy'] + [states_dict[step_idx + 1]['q'][-1]], dtype=np.float32) - np.array(states_dict[step_idx]['objects_to_track']['EE']['xyz'] + states_dict[step_idx]['objects_to_track']['EE']['rpy'] + [states_dict[step_idx]['q'][-1]], dtype=np.float32),
                    'discount': 1,
                    'reward': 1,
                    'is_first': step_idx == 0,
                    'is_last': step_idx == (len(states_dict) - 1),
                    'is_terminal': step_idx == (len(states_dict) - 1),
                    'goal_object': name_conversion_dict[states_dict[step_idx]['goal_object']],
                    'action_inst': states_dict[step_idx]['action_inst'],
                    'language_instruction': lang_inst,
                    'language_embedding': self._embed([lang_inst])[0].numpy(),
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        dirs = os.listdir(path)
        all_dirs = []
        for data_dir in dirs:
            all_dirs = all_dirs + [ f.path for f in os.scandir(os.path.join(path, data_dir)) if f.is_dir() ]
        print(all_dirs)

        # for smallish datasets, use single-thread parsing
        for sample in all_dirs:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

