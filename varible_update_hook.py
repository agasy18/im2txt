import tensorflow as tf

class VaribleUpdateHook(tf.train.SessionRunHook):
    def __init__(self, chackpoint_filepattern, var_map, run_count=1):
        self.loader = tf.train.NewCheckpointReader(chackpoint_filepattern)
        self.var_map = var_map
        self.run_count = run_count
        
    
    def begin(self):
        self.assign_ops = []
        self.placeholder_map = {}
        variable_to_dtype_map = self.loader.get_variable_to_dtype_map()
        variable_to_shape_map = self.loader.get_variable_to_shape_map()
        for k, v in self.var_map.items():
            placeholder = tf.placeholder(variable_to_dtype_map[v], variable_to_shape_map[v])
            self.placeholder_map[placeholder] = self.loader.get_tensor(v)
            self.assign_ops.append(tf.assign(tf.get_default_graph().get_tensor_by_name(k+':0'), placeholder))
    
    def after_create_session(self, session, coord):
        if self.run_count > 0:
            print ("Loading")
            session.run(self.assign_ops, self.placeholder_map)
            self.run_count -= 1
        del self.placeholder_map
        del self.assign_ops