# 输出tensorflow的graph
detection_pb = './checkpoint/ICDAR_0.7.pb'
with tf.gfile.FastGFile(os.path.join(detection_pb), 'rb') as f:
    # 使用tf.GraphDef()定义一个空的Graph
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # Imports the graph from graph_def into the current default Graph.
    tf.import_graph_def(graph_def, name='')


tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name,'\n')
