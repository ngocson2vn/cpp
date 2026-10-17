"""
Python >= 3.7.3
TensorFlow 2.5.0
"""


MODEL_DIR = f"./"

import os
import sys
import json

cwd = os.getcwd()
if cwd in sys.path:
  sys.path.remove(cwd)
print(sys.path)

if not os.path.exists(MODEL_DIR):
  os.makedirs(MODEL_DIR, exist_ok=True)

import tensorflow._api.v2.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()

graph_file_path = "{}/graph".format(MODEL_DIR)

inputs = []
float_inputs = []
gather_group1 = []

# create the input placeholder
N = 3
input = tf.placeholder(tf.float32, shape=[None, 8, 1024])
inputs.append(f"{input.op.name}:0")
const = tf.constant(5.000000e-01, dtype=tf.float32)
mul = tf.multiply(const, input)
relu = tf.maximum(mul, 0)
output1 = tf.identity(relu, name="Output_1")

fetch_list = [output1.op.name]

# launch the graph in a session
batch_size = 128
with tf.Session() as sess:
  # # create the dictionary:
  d = {}
  d[inputs[0]] = np.random.rand(16, 8, 1024)

  # # feed it to placeholder a via the dict 
  # print(sess.run(C, feed_dict=d))
  results = sess.run([f"{name}:0" for name in fetch_list], feed_dict=d)
  feed_dict = {}
  ndarr = []
  for k, v in d.items():
    ndarr = v
    feed_dict[k] = {"shape": v.shape, "dtype": v.dtype, "ndarr": ndarr}
  infer_input_file = f"{MODEL_DIR}/input.json"
  with open(infer_input_file, "w") as fw:
    fw.write(json.dumps({"feeds": feed_dict, "fetches": fetch_list}, indent=2, default=str))
    print(infer_input_file)
  for idx, fetch in enumerate(fetch_list):
    print(f"{fetch}:", results[idx])

graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_pbtxt_file = graph_file_path + ".pbtxt"
graph_writer = open(graph_pbtxt_file, "w")
graph_writer.write(str(graph_def))
graph_writer.close()
print(graph_pbtxt_file)
with tf.gfile.GFile(graph_file_path, 'wb') as f:
  data = graph_def.SerializeToString()
  f.write(data)
print(graph_file_path)

graph_pb_file = graph_file_path + ".pb"
with tf.gfile.GFile(graph_pb_file, 'wb') as f:
  data = graph_def.SerializeToString()
  f.write(data)
print(graph_pb_file)
