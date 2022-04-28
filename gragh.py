# 输出tensorflow的graph
detection_pb = './checkpoint/ICDAR_0.7.pb'
with tf.gfile.FastGFile(os.path.join(detection_pb), 'rb') as f:
    # 使用tf.GraphDef()定义一个空的Graph
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # Imports the graph from graph_def into the current default Graph.
    tf.import_graph_def(graph_def, name='')


# pb模型文件节点信息保存到log文件夹内，用tensorboard查看
with tf.Session() as sess:
    with tf.gfile.GFile(os.path.join(detection_pb), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
        train_writer = tf.summary.FileWriter("./log")
        train_writer.add_graph(sess.graph)
        train_writer.flush()
        train_writer.close()
    
    
    
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name,'\n')
