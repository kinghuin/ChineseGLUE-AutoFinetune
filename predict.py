#coding:utf-8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Finetuning on classification task """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import numpy as np
import os
import time
import paddle
import paddle.fluid as fluid
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--dataset", type=str, default="chnsenticorp", help="The choice of dataset")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Download dataset and use ClassifyReader to read dataset
    if args.dataset.lower() == 'inews':
        dataset = hub.dataset.INews()
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
        batch_size = 4
        max_seq_len = 512
        num_epoch = 3
    elif args.dataset.lower().startswith("xnli"):
        dataset = hub.dataset.XNLI(language="zh")
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
        batch_size = 32
        max_seq_len = 128
        num_epoch = 2
    elif args.dataset.lower() == "lcqmc":
        dataset = hub.dataset.LCQMC()
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
        batch_size = 16
        max_seq_len = 128
        num_epoch = 3
    elif args.dataset.lower() == "tnews":
        dataset = hub.dataset.TNews()
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
        batch_size = 16
        max_seq_len = 128
        num_epoch = 3
    else:
        raise ValueError("%s dataset is not defined" % args.dataset)

    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=max_seq_len)

    reader = hub.reader.ClassifyReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=max_seq_len,
        use_task_id=False)
    pooled_output = outputs["pooled_output"]

    # Setup feed list for data feeder
    # Must feed all the tensor of ERNIE's module need
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_data_parallel=False,
        use_pyreader=False,
        use_cuda=True,
        batch_size=batch_size,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

    # Define a classfication finetune task by PaddleHub's API
    cls_task = hub.TextClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config,
        metrics_choices=metrics_choices)

    # Data to be prdicted
    data = [[d.text_a, d.text_b] for d in dataset.get_test_examples()]

    index = 0
    run_states = cls_task.predict(data=data)
    with open(args.dataset.lower() + "_predict.txt", "w") as fout:
        results = [run_state.run_results for run_state in run_states]
        for batch_result in results:
            # get predict index
            batch_result = np.argmax(batch_result, axis=2)[0]
            for result in batch_result:
                label = dataset.get_labels()[int(result)]
                if index < 3:
                    print("%s\tpredict= %s" % (data[index][0], label))
                fout.write(label + "\n")
                index += 1
