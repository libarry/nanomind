# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re
from typing import Dict, List, Union, Type, Pattern, Tuple
from collections import OrderedDict
from ..models.model_utils import (
    StructurePair, 
    TransformerStructurePairVisitor
)


def match_pattern(pair_name: str, pattern: Union[str, Pattern]):
    if isinstance(pattern, str):
        return pair_name.startswith(pattern)
    else: 
        return pattern.match(pair_name)


class QuantizerMapper:
    """
    管理量化器映射表的类，根据pattern匹配为结构对分配对应的quantizer
    """
    def __init__(self):
        # 存储 pattern -> visitor 的映射
        self.pattern_map: Dict[Union[str, Pattern], TransformerStructurePairVisitor] = OrderedDict()
        # 存储 pair_name -> quantizer 的有序映射结果
        self.pair_quantizer_map: OrderedDict[str, TransformerStructurePairVisitor] = OrderedDict()
    
    def register_pattern(self, pattern: Union[str, Pattern], visitor: TransformerStructurePairVisitor):
        self.pattern_map[pattern] = visitor
        return self

    def apply_quantizer(self, pairs_dict: Dict[str, List[StructurePair]]):
        self.pair_quantizer_map.clear()
        pairs_list = []
        for pairs in pairs_dict.values():
            for pair in pairs:
                pairs_list.append(pair)

        for pattern, visitor in self.pattern_map.items():
            for pair in pairs_list:
                pair_str = str(pair)
                if match_pattern(pair_str, pattern):
                    self.pair_quantizer_map[pair_str] = visitor
                    pair.accept(visitor)
    
    def get_quantizer_for_pair(self, pair: Union[StructurePair, str]) -> Union[TransformerStructurePairVisitor, None]:
        pair_str = str(pair)
        return self.pair_quantizer_map.get(pair_str)
    
    def to_org_mode(self, prefix=""):
        for visitor in set(self.pair_quantizer_map.values()):
            visitor.to_org_mode(prefix)
    
    def to_calib_mode(self, prefix=""):
        for visitor in set(self.pair_quantizer_map.values()):
            if hasattr(visitor, 'to_calib_mode'):
                visitor.to_calib_mode(prefix)
    
    def to_eval_mode(self, prefix="", quant_weight=True):
        for visitor in set(self.pair_quantizer_map.values()):
            if hasattr(visitor, 'to_eval_mode'):
                visitor.to_eval_mode(prefix=prefix, quant_weight=quant_weight) 