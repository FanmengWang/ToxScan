# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
    RawArrayDataset,
    FromNumpyDataset,
)
from ToxScan.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    data_utils,
    PrependAndAppend2DDataset,
)

from ToxScan.data.tta_dataset import TTADataset
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)

@register_task("ToxScan")
class ToxScanTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=2,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen ",
        )


    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pad_idx = self.dictionary.pad()

        self.atom_charge_pad_idx = 0
        self.atom_H_pad_idx = 6
        self.bond_pad_idx = 8
        
        self.atom_charge_mask_idx = 9
        self.atom_H_mask_idx = 10
        self.bond_mask_idx = 12
        
        self.target_pad_idx = -10000.0
        self.classification_target_pad_idx = -10000
        self.regression_target_pad_idx = -10000.0
        
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif self.args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True
            

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)


    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        
        if split == "train":
            fingerprint_dataset = KeyDataset(dataset, "fingerprint")
            tgt_dataset = KeyDataset(dataset, "target")
            classification_tgt_dataset = KeyDataset(dataset, "classification_target")
            regression_tgt_dataset = KeyDataset(dataset, "regression_target")
            smi_dataset = KeyDataset(dataset, "data")
            sample_dataset = ConformerSampleDataset(
                dataset, self.args.seed, "atoms", "coordinates", "atom_feature", "bond_feature"
            )
            dataset = AtomTypeDataset(dataset, sample_dataset)
        
        else:
            dataset = TTADataset(
                dataset, self.args.seed, "atoms", "coordinates", 
                "atom_feature", "bond_feature", "fingerprint", 
                "target", "classification_target", "regression_target", self.args.conf_size
            )
            dataset = AtomTypeDataset(dataset, dataset)
            fingerprint_dataset = KeyDataset(dataset, "fingerprint")
            tgt_dataset = KeyDataset(dataset, "target")
            classification_tgt_dataset = KeyDataset(dataset, "classification_target")
            regression_tgt_dataset = KeyDataset(dataset, "regression_target")
            smi_dataset = KeyDataset(dataset, "data")

        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            "atom_feature", 
            "bond_feature",
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        )
        
        dataset = CroppingDataset(
            dataset, 
            self.seed, 
            "atoms", 
            "coordinates", 
            "atom_feature", 
            "bond_feature", 
            self.args.max_atoms
        )
        
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coordinates")
        
        atom_feature_dataset0 = KeyDataset(dataset, "atom_feature0")
        atom_feature_dataset1 = KeyDataset(dataset, "atom_feature1")
        bond_feature_dataset0 = KeyDataset(dataset, 'bond_feature0')
        bond_feature_dataset1 = KeyDataset(dataset, 'bond_feature1')
        bond_feature_dataset2 = KeyDataset(dataset, 'bond_feature2')

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)
        
        atom_feature_dataset0 = FromNumpyDataset(atom_feature_dataset0)
        atom_feature_dataset0 = PrependAndAppend(atom_feature_dataset0, self.atom_charge_mask_idx - 2 , self.atom_charge_mask_idx - 1)
        atom_feature_dataset1 = FromNumpyDataset(atom_feature_dataset1)
        atom_feature_dataset1 = PrependAndAppend(atom_feature_dataset1, self.atom_H_mask_idx - 2, self.atom_H_mask_idx - 1)
        
        bond_feature_dataset0 = PrependAndAppend2DDataset(FromNumpyDataset(bond_feature_dataset0), self.bond_mask_idx - 1)
        bond_feature_dataset1 = PrependAndAppend2DDataset(FromNumpyDataset(bond_feature_dataset1), self.bond_mask_idx - 1)
        bond_feature_dataset2 = PrependAndAppend2DDataset(FromNumpyDataset(bond_feature_dataset2), self.bond_mask_idx - 1)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "src_atom_feature0": RightPadDataset(
                        atom_feature_dataset0,
                        pad_idx=self.atom_charge_pad_idx,
                    ),
                    "src_atom_feature1": RightPadDataset(
                        atom_feature_dataset1,
                        pad_idx=self.atom_H_pad_idx,
                    ),
                    "src_bond_feature0": RightPadDataset2D(
                        bond_feature_dataset0,
                        pad_idx=self.bond_pad_idx,
                    ),
                    "src_bond_feature1": RightPadDataset2D(
                        bond_feature_dataset1,
                        pad_idx=self.bond_pad_idx,
                    ),
                    "src_bond_feature2": RightPadDataset2D(
                        bond_feature_dataset2,
                        pad_idx=self.bond_pad_idx,
                    ),
                    "fingerprint": RawLabelDataset(fingerprint_dataset)
                },
                
                "target": {
                    "target": RawLabelDataset(tgt_dataset),
                    "classification_target": RawLabelDataset(classification_tgt_dataset),
                    "regression_target": RawLabelDataset(regression_tgt_dataset),
                },
                
                "smi_name": RawArrayDataset(smi_dataset),
            },
        )
        
        if not self.args.no_shuffle and split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset


    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        return model
