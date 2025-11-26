# Copyright 2024 HuggingFace Inc..
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class HyperAdapterArguments:
    """
    The subset of arguments related to adapter training.

    Args:
        train_adapter (bool): Whether to train an adapter instead of the full model.
        load_adapter (str): Pre-trained adapter module to be loaded from Hub.
        adapter_config (str): Adapter configuration. Either a config string or a path to a file.
        load_lang_adapter (str): Pre-trained language adapter module to be loaded from Hub.
        lang_adapter_config (str): Language adapter configuration. Either an identifier or a path to a file.
    """

    hyper_adapters: bool = field(default=False, metadata={"help": "If use hyper adapter."})
    freeze_pretrained: bool = field(default=False, metadata={"help": "Freeze the parameters of the main network. "
                                 "Applicable to experiments for (hyper-)adapters finetuning."})
    hyper_adapters_bottle: Optional[int] = field(
        default=128, metadata={"help": "The dimensionality of the generated adapter bottleneck representation."}
    )
    hyper_adapters_hidden_dim: Optional[int] = field(
        default=64, metadata={"help": "The dimensionality of the hyper-network hidden representations."}
    )
    hyper_adapters_hidden_layers: Optional[int] = field(
        default=2, metadata={"help": "The number of hidden layers in the hyper-network."}
    )
    hyper_adapters_lang_embed_dim: Optional[int] = field(
        default=50, metadata={"help": "The dimensionality of the language embeddings of the hyper-network."}
    )
    hyper_adapters_layer_embed_dim: Optional[int] = field(
        default=50, metadata={"help": "The dimensionality of the layer embeddings of the hyper-network."}
    )
    hyper_adapters_dropout: Optional[float] = field(
        default=0.0, metadata={"help": "Dropout used in the hyper-network."}
    )
    per_lang_pair_batch_size: Optional[int] = field(
        default=1, metadata={
            "help": "batch size of each language pair"
        },
    )
    hyper_adapters_activation_fn: Optional[str] = field(
        default="relu", metadata={"help": "Activation function for the hyper-network."}
    )
    hyper_adapters_lang_embed_tied: bool = field(
        default=False,
        metadata={"help": "Use a share embedding matrix for the source and target language embeddings."},
    )
    hyper_adapters_layernorm_input: bool = field(
        default=False,
        metadata={"help": "Apply layer normalization to the hyper-network input (lang+layer embeddings)."},
    )
    hyper_adapters_layernorm_output: bool = field(
        default=False,
        metadata={"help": "Apply layer normalization to the hyper-network output (before weight generation projections)."},
    )
    hyper_adapters_generate_layernorm: bool = field(
        default=False,
        metadata={"help": "Generate from the hyper-network the LayerNorm "
                                 "parameters for each generated adapter layer."},
    )
    hyper_adapters_no_rescale: bool = field(
        default=False,
        metadata={"help": "Disable the weight rescaling in the hyper-network."},
    )
    hyper_adapters_init: Optional[str] = field(
        default="fairseq", metadata={"help": "Initialization method for the weights of the hyper-network. choices=['default', 'hyper']"}
    )
    language_list: Optional[str] = field(
        default="english,russian,indonesian,urdu", metadata={"help": "All language in the dataset."}
    )
    hyper_predict: bool = field(
        default=False,
        metadata={"help": "Do predict."},
    )
    hyper_src_lang: Optional[int] = field(
        default=100, metadata={
            "help": "batch size of each language pair"
        },
    )
    hyper_tgt_lang: Optional[int] = field(
        default=100, metadata={
            "help": "batch size of each language pair"
        },
    )
    hyper_classification: bool = field(
        default=False,
        metadata={"help": "Do hyper param classification."},
    )
    hyper_classification_loss_ratio: Optional[float] = field(
        default=0.1, metadata={
            "help": "The ratio of hyper classification loss."
        },
    )
    use_hyper: bool = field(
        default=False,
        metadata={"help": "If use hyper-adapter."},
    )
    use_img: bool = field(
        default=False,
        metadata={"help": "If use img."},
    )
    lora_train_hyper: bool = field(
        default=False,
        metadata={"help": "If use lora train hyper."},
    )
    img_path: Optional[str] = field(
        default="/m2m_images_feature/individual_img", metadata={"help": "Directory for storing img files."}
    )
    
         


         