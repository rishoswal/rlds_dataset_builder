from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class RealBimanual(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'timestamp': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='Timestamp of the observation.'
                        ),
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'left_wrist_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Left wrist camera RGB observation.',
                        ),
                        'right_wrist_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Right wrist camera RGB observation.',
                        ),
                        'left_gripper_position': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='Left gripper position.'
                        ),
                        'right_gripper_position': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='Right gripper position.'
                        ),
                        'state': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float64,
                            doc='Robot joint state (7D left arm + 7D right arm).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float64,
                        doc='Robot arm action, (7D left arm + 7D right arm)',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
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
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
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
            'train': self._generate_examples(path='/home/prior/rishoswal/bimanual_oculus/orange_apple_in_bowl/*'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            print(list(data.keys()))

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(len(data["timestamp"])):
                # compute Kona language embedding
                language_embedding = self._embed([data['language_instruction'][0]])[0].numpy()

                episode.append({
                    'observation': {
                        'timestamp': data['timestamp'][i],
                        'image': data['camera_image_3'][i],
                        'left_wrist_image': data['camera_image_1'][i],
                        'right_wrist_image': data['camera_image_2'][i],
                        'left_gripper_position': data['left_gripper_position'][i],
                        'right_gripper_position': data['right_gripper_position'][i],
                        'state': np.concatenate([data['left_joint_positions'][i], data['right_joint_positions'][i]]),
                    },
                    'action': np.concatenate([data['left_delta_action'][i], data['right_delta_action'][i]]),
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': data['language_instruction'][0],
                    'language_embedding': language_embedding
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
        episode_paths = glob.glob(path)
        # episode_paths = ["/home/prior/rishoswal/bimanual_oculus/demos/20250327T161224_0_273.npz"]

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

