
import os
import sys

while True:
    model_path = input("请输入模型路径:").strip()

    output_dir = input("请输入onnx导出路径:").strip()

    onnx_name = input("请输入导出的onnx模型名称:").strip()

    model_class_name = input("请输入模型类名(例如:TFBertModel):").strip()

    input_dim_shape = input("请输入输入参数shape(例如:1,300):").strip()

    pycode = f"""

# Whether allow overwrite existing script or model.
import os
import time

from transformers import {model_class_name}, BertTokenizer


model_path = r'{model_path}'

output_dir = r'{output_dir}'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("111")

# Number of runs to get average latency.
total_runs = 100

# Max sequence length for the export model
max_sequence_length = 300

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU') # Disable GPU for fair comparison

tokenizer = BertTokenizer.from_pretrained(model_path)

model = TFBertModel.from_pretrained(model_path)

model._saved_model_inputs_spec = None

import numpy

question, text = "What is ONNX Runtime?", "ONNX Runtime is a performance-focused inference engine for ONNX models."
# Pad to max length is needed. Otherwise, position embedding might be truncated by constant folding.

inputs = tokenizer.encode_plus(question,add_special_tokens=True,return_tensors='tf',max_length=max_sequence_length,truncation=True)
output = model(inputs)
start_scores, end_scores = output.last_hidden_state, output.pooler_output

num_tokens = len(inputs["input_ids"][0])

import tf2onnx

tf2onnx.logging.set_level(tf2onnx.logging.ERROR)

opset_version = 13
use_external_data_format = False

specs = []
for name, value in inputs.items():
    # dims = [None] * len(value.shape)
    dims = [{input_dim_shape}]
    specs.append(tf.TensorSpec(tuple(dims), value.dtype, name=name))


output_model_path =  os.path.join(output_dir, '{onnx_name}.onnx')


start = time.time()

_,_ = tf2onnx.convert.from_keras(model,
                          input_signature=tuple(specs),
                          opset=opset_version,
                          large_model=use_external_data_format,
                          output_path=output_model_path)


print("tf2onnx run time = {{}} s".format(format(time.time() - start, '.2f')))
"""

    with open("test.py","w",encoding="utf-8") as wf:
        wf.write(pycode)

    dirpath = os.path.dirname(os.path.realpath(sys.argv[0]))
    os.system(f"{dirpath}/cpu_env/python.exe {dirpath}/test.py")
    print("导出完毕!")
    iscontinue = input("是否继续(y|n):")
    if iscontinue == "y":
        continue
    else:
        exit(0)