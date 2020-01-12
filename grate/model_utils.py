import tensorflow as tf

class Model_utils(object):
    @staticmethod
    def placeholders(placeholder_dict):
        placeholders = {}
        for name, data in placeholder_dict.items():
            placeholders[name] = tf.placeholder(dtype = tf.float32,shape=data)
        return placeholders
    
    @staticmethod
    def pos_and_norm(adj_matrix, feature_matrix):
    
        num_nodes    = adj_matrix.shape[0]
        num_features = feature_matrix.shape[1]
        
        pos_weight = float(adj_matrix.shape[0] * adj_matrix.shape[0] - adj_matrix.sum()) / adj_matrix.sum()
        norm = adj_matrix.shape[0] * adj_matrix.shape[0] / float((adj_matrix.shape[0] * adj_matrix.shape[0] - adj_matrix.sum()) * 2)
        
        return {
                'num_nodes': num_nodes, 
                'num_features': num_features, 
                'pos_weight': pos_weight, 
                'norm': norm
                }

    
    