import tensorflow as tf

class VaribleUpdateHook(tf.train.SessionRunHook):
    def __init__(self, chackpoint_filepattern, var_map):
        self.loader = tf.train.NewCheckpointReader(chackpoint_filepattern)
        self.var_map = var_map
        self.assign_ops = []
        self.placeholder_map = {}
    
    def begin(self):
        variable_to_dtype_map = self.loader.get_variable_to_dtype_map()
        variable_to_shape_map = self.loader.get_variable_to_shape_map()
        for k, v in self.var_map.items():
            placeholder = tf.placeholder(variable_to_dtype_map[v], variable_to_shape_map[v])
            self.placeholder_map[placeholder] = self.loader.get_tensor(v)
            self.assign_ops.append(tf.assign(tf.get_default_graph().get_tensor_by_name(k+':0'), placeholder))
    
    def after_create_session(self, session, coord):
        session.run(self.assign_ops, self.placeholder_map)