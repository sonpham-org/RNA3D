# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# model configs for inference and training,
# such as: rnapro-base, rnapro-mini, rnapro-tiny, rnapro-constraint.

model_configs = {
    "rnapro_base": {
        "model": {"N_cycle": 10},
        "sample_diffusion": {
            "N_step": 200,
        },  # the default setting for base model
    },
    "rnapro_mini": {
        "sample_diffusion": {
            "gamma0": 0,
            "step_scale_eta": 1.0,
            "N_step": 5,
        },  # the default setting for mini model
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 16,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "load_strict": False,  # For inference, it should be True.
    },
    "rnapro_tiny": {
        "sample_diffusion": {
            "gamma0": 0,
            "step_scale_eta": 1.0,
            "N_step": 5,
        },  # the default setting for tiny model
        "model": {
            "N_cycle": 4,
            "msa_module": {
                "n_blocks": 1,
            },
            "pairformer": {
                "n_blocks": 8,
            },
            "diffusion_module": {
                "atom_encoder": {
                    "n_blocks": 1,
                },
                "transformer": {
                    "n_blocks": 8,
                },
                "atom_decoder": {
                    "n_blocks": 1,
                },
            },
        },
        "load_strict": False,  # For inference, it should be True.
    },
}
