import tensorflow as tf
from utils.classifier import Classifier
from trainer import Trainer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

from optimizer import OptimizerVAEt

from input_data import load_data

import numpy as np
from utils import process
import scipy.sparse as sp
tf.reset_default_graph()

dataset_str = 'citeseer'

# Load data
adj, features = load_data(dataset_str)

hyu = features.toarray()

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train
adj_norm = preprocess_graph(adj)

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

def encoder(A, H, w, aw0,aw1,attention_no):
    H = tf.matmul(H, w)
    C = graph_attention_layer(A, H, aw0,aw1,attention_no)
    return tf.sparse_tensor_dense_matmul(C, H) , C



def graph_attention_layer(A, M, v0,v1, layer):
    

    with tf.variable_scope("layer_%s"% layer):
        
        
        f1 = tf.matmul(M, v0)
        f1 = A * f1
        f2 = tf.matmul(M, v1)
        f2 = A * tf.transpose(f2, [1, 0])
        logits = tf.sparse_add(f1, f2)

        unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                     values=tf.nn.sigmoid(logits.values),
                                     dense_shape=logits.dense_shape)
        
        
        attentions = tf.sparse_softmax(unnormalized_attentions)

        attentions = tf.SparseTensor(indices=attentions.indices,
                                     values=attentions.values,
                                     dense_shape=attentions.dense_shape)
        
        

        return attentions
        
Art = tf.sparse_placeholder(dtype=tf.float32) #adj matrix node node
Xrt = tf.placeholder(dtype=tf.float32)        #feature matrix node feature
adj_origr = tf.sparse_placeholder(tf.float32)

feature_no_of = 3703


weight_layer_0 = tf.get_variable(name='weight_0',shape=[feature_no_of,512],
                                                  initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
weight_layer_1 = tf.get_variable(name='weight_1',shape=[512,512],
                                                  initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
weight_atten_00 = tf.get_variable(name='weight_attn_0',shape=[512,1],
                                                  initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
weight_atten_01 = tf.get_variable(name='weight_attn_01',shape=[512,1],
                                                  initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
weight_atten_10 = tf.get_variable(name='weight_attn_1',shape=[512,1],
                                                  initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
weight_atten_11 = tf.get_variable(name='weight_attn_10',shape=[512,1],
                                                  initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)


#encoder part

node_no = 3327

with tf.variable_scope("encoder"):
    
    H = Xrt
    
    first_layer, attention_weights_0   = encoder(Art, H, weight_layer_0, weight_atten_00,weight_atten_01,0)
    z_mean, attention_weights_1        = encoder(Art, first_layer ,weight_layer_1,weight_atten_10, weight_atten_11,1)
    z_log_stf, attention_weights_1s    = encoder(Art, first_layer ,weight_layer_1,weight_atten_10, weight_atten_11,2)
    
    

    z = z_mean + tf.random_normal([node_no,512]) * tf.exp(z_log_stf)
    

reconstructions = InnerProductDecoder(weight_layer_1)(z)

opt = OptimizerVAEt(reconstructions,tf.reshape(tf.sparse_tensor_to_dense(adj_origr, False), [-1]),num_nodes,
                           pos_weight,
                           norm,z_log_stf,z_mean)
                           
                           
import time

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        emb = sess.run(z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Train model
for epoch in range(200):

    t = time.time()
    # Construct feed dictionary
    
    
    feed_dict = {Art: adj_norm, Xrt: hyu , adj_origr : adj_label}


    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]
    
    print(len(val_edges),len(val_edges_false))

    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    val_roc_score.append(roc_curr)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")

roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
