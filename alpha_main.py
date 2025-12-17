# ALPHA: Attribute-augmented Lightweight Privacy-preserving Hybrid Attentive network for Tourism Recommendation
# Tensorflow v1 implementation (tensorflow-gpu=1.13.0)
# Author: Chenxi Ma (chenxim1998@outlook.com)

from time import time
import random
import os
import hashlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import logging

np.seterr(all='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 2025
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)


################## Part 1 超参数配置 ##################
class ParamConfig:
    def __init__(self):
        '''
           block 1: the hyper parameters for model training
        '''
        self.learning_rate = 0.001
        self.dropout_rate = 0.1
        self.batch_size = 256
        self.num_epochs = 60
        self.eval_verbose = 10
        self.fast_running = False
        self.fast_ratio = 0.5

        '''
            block 2: the hyper parameters for ALPHA model
        '''
        self.embedding_size = 16
        self.num_gcn_layers = 2  # α
        self.num_attention_heads = 2  # H
        self.residual_weight = 1.0  # β
        self.gpu_index = '0'

        '''
            block 3: dataset and privacy settings
        '''
        self.dataset = "TripAdvisor-Hotel"  # TripAdvisor-Hotel, Yelp-Travel, Qunar-Flight
        self.train_path = "../Data/" + self.dataset + "/train_data.txt"
        self.test_path = "../Data/" + self.dataset + "/test_data.txt"
        self.attr_path = "../Data/" + self.dataset + "/attr_data.txt"
        self.item_path = "../Data/" + self.dataset + "/item_dict.txt"
        self.user_path = "../Data/" + self.dataset + "/user_dict.txt"
        self.check_points = "../check_points/" + self.dataset + ".ckpt"

        self.salt = b"ALPHA_PRIVACY_SALT_2025"  # Random salt for pseudonymization (we utilize a random text here)
        self.pseudonymize = True    # (for testing)

        # PGCN settings
        self.num_folded = self.embedding_size


################## Part 2 数据集加载部分配置 ##################
def load_dict(dict_path):
    dict_output = {}
    with open(dict_path, 'r') as file_object:
        elements = file_object.readlines()
    for dict_element in elements:
        dict_element = dict_element.strip().split('\t')
        dict_output[dict_element[1]] = int(dict_element[0])
    return dict_output


def pseudonymize_user(user_id, salt):
    """Pseudonymize user ID using SHA-256 hashing with random salt"""
    hash_input = str(user_id).encode() + salt
    hashed = hashlib.sha256(hash_input).hexdigest()
    # Convert hex digest to integer for use as ID
    return int(hashed[:8], 16) % (2 ** 31 - 1)


def load_attribute_data(attr_path, user_dict, salt, pseudonymize=True):
    """Load attribute Data and create profile knowledge graph"""
    with open(attr_path, 'r') as f:
        lines = f.readlines()

    user_attr_map = {}
    attr_set = set()

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue

        user_id = parts[0]
        if user_id not in user_dict:
            continue

        user_idx = user_dict[user_id]
        if pseudonymize:
            user_idx = pseudonymize_user(user_idx, salt)

        attributes = parts[1].split(',')
        user_attr_map[user_idx] = []

        for attr in attributes:
            if attr:
                attr_set.add(attr)
                user_attr_map[user_idx].append(attr)

    # Create attribute dictionary
    attr_dict = {attr: idx for idx, attr in enumerate(sorted(list(attr_set)))}

    return user_attr_map, attr_dict


def config_input(data_path, item_dict, user_dict):
    with open(data_path, 'r') as data_file:
        data = []
        lines = data_file.readlines()
        for line in lines:
            temp = []
            item = []
            position = []
            pos_index = 0
            length = 0
            line_split = line.strip().split('\t')
            item.append(user_dict[line_split[0]])
            for index in line_split[1:-1]:
                item.append(item_dict[index])
                position.append(pos_index)
                pos_index += 1
                length += 1
            temp.append(item)  # user_id & item_id
            temp.append(position)  # positional information for prompt-tuning
            temp.append(length)  # length of the item
            temp.append(item_dict[line_split[-1]])  # target_item
            data.append(temp)
    return data


def construct_bipartite_graph(data, num_users, num_items):
    user_item_pairs = []

    for record in data:
        item = record[0]
        user = int(item[0])  # user_id
        items = [int(i) for i in item[1:]]  # user's interacted items

        for item in items:
            user_item_pairs.append([user, item])

    # Create adjacency matrix for bipartite graph
    rows = [pair[0] for pair in user_item_pairs]
    cols = [pair[1] + num_users for pair in user_item_pairs]  # offset for items
    values = [1.0] * len(rows)

    # Also add reverse edges
    rows_rev = cols
    cols_rev = rows
    values_rev = values

    # Combine
    all_rows = rows + rows_rev
    all_cols = cols + cols_rev
    all_values = values + values_rev

    adj_matrix = sp.coo_matrix((all_values, (all_rows, all_cols)),
                               shape=(num_users + num_items, num_users + num_items))

    return adj_matrix


def construct_knowledge_graph(user_attr_map, attr_dict, num_users, num_items):
    user_nodes = []
    attr_nodes = []
    values = []

    for user_idx, attrs in user_attr_map.items():
        for attr in attrs:
            attr_idx = attr_dict[attr]
            # User to attribute edge
            user_nodes.append(user_idx)
            attr_nodes.append(num_users + num_items + attr_idx)
            values.append(1.0)
            # Attribute to user edge
            attr_nodes.append(num_users + num_items + attr_idx)
            user_nodes.append(user_idx)
            values.append(1.0)

    # Add self-loops for attribute nodes
    total_nodes = num_users + num_items + len(attr_dict)
    for i in range(total_nodes):
        user_nodes.append(i)
        attr_nodes.append(i)
        values.append(1.0)

    adj_matrix = sp.coo_matrix((values, (user_nodes, attr_nodes)),
                               shape=(total_nodes, total_nodes))

    return adj_matrix, len(attr_dict)


def normalize_adjacency_matrix(adj_matrix):
    rowsum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized_adj = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_adj.tocoo()


def load_batches(batch, padding_num):
    user, item, position, length, target = [], [], [], [], []
    for data_index in batch:
        length.append(data_index[2])
    max_length = max(length)

    i = 0
    for data_index in range(len(batch)):
        user.append(batch[data_index][0][0])
        item.append(batch[data_index][0][1:] + [padding_num] * (max_length - length[i]))
        position.append(batch[data_index][1][0:] + [padding_num] * (max_length - length[i]))
        target.append(batch[data_index][3])
        i += 1

    return np.array(user), np.array(item), np.array(position), np.array(length).reshape(len(length), 1), np.array(
        target)


def generate_batches(input_data, batch_size, padding_num, is_train):
    user_all, item_all, position_all, length_all, target_all = [], [], [], [], []
    num_batches = int(len(input_data) / batch_size)  ####

    if is_train is True:
        random.shuffle(input_data)

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        batch = input_data[start_index:start_index + batch_size]
        user, item, position, length, target = load_batches(batch=batch, padding_num=padding_num)

        user_all.append(user)
        item_all.append(item)
        position_all.append(position)
        length_all.append(length)
        target_all.append(target)

    return list((user_all, item_all, position_all, length_all, target_all, num_batches))


################## Part 3 ALPHA模型部分 ##################

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def get_inputs():
    user = tf.placeholder(dtype=tf.int32, shape=[None, ], name='user')
    item = tf.placeholder(dtype=tf.int32, shape=[None, None], name='item')
    position = tf.placeholder(dtype=tf.int32, shape=[None, None], name="position")
    length = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='length')
    target = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')
    return user, item, position, length, target, learning_rate, dropout_rate


def loss_calculation(target, pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=pred)
    loss_mean = tf.reduce_mean(loss, name='loss_mean')
    return loss_mean


def optimizer(loss, learning_rate):
    basic_op = tf.train.AdamOptimizer(learning_rate)
    gradients = basic_op.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                        grad is not None]
    model_op = basic_op.apply_gradients(capped_gradients)
    return model_op


class ParallelGraphConvolutionNetwork:
    def __init__(self, num_users, num_items, num_attrs,
                 bipartite_adj, knowledge_adj,
                 embedding_size, num_layers):
        self.num_users = num_users
        self.num_items = num_items
        self.num_attrs = num_attrs
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        # Normalize adjacency matrices
        self.bipartite_adj_norm = normalize_adjacency_matrix(bipartite_adj)
        self.knowledge_adj_norm = normalize_adjacency_matrix(knowledge_adj)

    def build(self):
        with tf.variable_scope('PGCN_embeddings'):
            user_embeddings = tf.get_variable(
                'user_embeddings',
                shape=[self.num_users, self.embedding_size],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            item_embeddings = tf.get_variable(
                'item_embeddings',
                shape=[self.num_items, self.embedding_size],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            attr_embeddings = tf.get_variable(
                'attr_embeddings',
                shape=[self.num_attrs, self.embedding_size],
                initializer=tf.contrib.layers.xavier_initializer()
            )

        # Bipartite graph embeddings (travelers and items)
        bipartite_embeddings = tf.concat([user_embeddings, item_embeddings], axis=0)

        # Knowledge graph embeddings (pseudonymized travelers and attributes)
        # Pseudonymized travelers use same embeddings but different indices
        knowledge_embeddings = tf.concat([user_embeddings, attr_embeddings], axis=0)

        # Convert adjacency matrices to sparse tensors
        bipartite_adj_tensor = _convert_sp_mat_to_sp_tensor(self.bipartite_adj_norm)
        knowledge_adj_tensor = _convert_sp_mat_to_sp_tensor(self.knowledge_adj_norm)

        # Layer-wise propagation
        bipartite_all_embeddings = [bipartite_embeddings]
        knowledge_all_embeddings = [knowledge_embeddings]

        for layer in range(self.num_layers):
            # Bipartite graph convolution
            bipartite_embeddings = tf.sparse_tensor_dense_matmul(bipartite_adj_tensor, bipartite_embeddings)
            bipartite_all_embeddings.append(bipartite_embeddings)

            # Knowledge graph convolution
            knowledge_embeddings = tf.sparse_tensor_dense_matmul(knowledge_adj_tensor, knowledge_embeddings)
            knowledge_all_embeddings.append(knowledge_embeddings)

        # Aggregate layer-wise embeddings (Eqn. 7 and 10 in paper)
        bipartite_final = tf.reduce_mean(tf.stack(bipartite_all_embeddings, axis=1), axis=1)
        knowledge_final = tf.reduce_mean(tf.stack(knowledge_all_embeddings, axis=1), axis=1)

        # Split embeddings
        user_emb_final, item_emb_final = tf.split(bipartite_final, [self.num_users, self.num_items], axis=0)
        user_attr_emb_final, attr_emb_final = tf.split(knowledge_final, [self.num_users, self.num_attrs], axis=0)

        return user_emb_final, item_emb_final, attr_emb_final


class DualChannelHybridAttentionNetwork:
    def __init__(self, embedding_size, num_heads, residual_weight):
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.residual_weight = residual_weight
        self.head_dim = embedding_size // num_heads

    def linear_attention(self, query, key, value, name="linear_attention"):
        """Linear attention mechanism (Eqn. 11-13 in paper)"""
        with tf.variable_scope(name):
            # Compute K^T * V first (outer product) for linear complexity
            kt_v = tf.matmul(key, value, transpose_a=True)  # [d, d]
            # Scale
            kt_v = kt_v / tf.sqrt(tf.cast(self.embedding_size, tf.float32))
            # Linear projection with query
            output = tf.matmul(query, kt_v)  # [N, d]
            return output

    def multi_head_linear_attention(self, query, key, value, name="multi_head_linear_attention"):
        """Multi-head linear attention"""
        with tf.variable_scope(name):
            outputs = []

            for h in range(self.num_heads):
                # Project to subspace for each head
                W_q = tf.get_variable(f'W_q_{h}', shape=[self.embedding_size, self.head_dim],
                                      initializer=tf.contrib.layers.xavier_initializer())
                W_k = tf.get_variable(f'W_k_{h}', shape=[self.embedding_size, self.head_dim],
                                      initializer=tf.contrib.layers.xavier_initializer())
                W_v = tf.get_variable(f'W_v_{h}', shape=[self.embedding_size, self.head_dim],
                                      initializer=tf.contrib.layers.xavier_initializer())

                Q_h = tf.matmul(query, W_q)
                K_h = tf.matmul(key, W_k)
                V_h = tf.matmul(value, W_v)

                # Linear attention in subspace
                kt_v_h = tf.matmul(K_h, V_h, transpose_a=True)  # [head_dim, head_dim]
                kt_v_h = kt_v_h / tf.sqrt(tf.cast(self.head_dim, tf.float32))

                output_h = tf.matmul(Q_h, kt_v_h)  # [N, head_dim]
                outputs.append(output_h)

            # Concatenate and project back
            concatenated = tf.concat(outputs, axis=1)  # [N, embedding_size]
            W_o = tf.get_variable('W_o', shape=[self.embedding_size, self.embedding_size],
                                  initializer=tf.contrib.layers.xavier_initializer())

            output = tf.matmul(concatenated, W_o)
            return output

    def traveler_channel(self, user_embeddings, attr_embeddings, name="traveler_channel"):
        """Traveler channel with query-controlled knowledge transfer (Eqn. 11-14)"""
        with tf.variable_scope(name):
            # Query: user embeddings, Key/Value: attribute embeddings
            refined_embeddings = self.multi_head_linear_attention(
                query=user_embeddings,
                key=attr_embeddings,
                value=attr_embeddings,
                name="traveler_attention"
            )

            # Residual connection and layer normalization (Eqn. 14)
            refined_embeddings = refined_embeddings * self.residual_weight
            output = tf.contrib.layers.layer_norm(user_embeddings + refined_embeddings)

            return output

    def item_channel(self, item_embeddings, name="item_channel"):
        """Item channel for capturing global item correlations (Eqn. 15-17)"""
        with tf.variable_scope(name):
            # Self-attention on items
            refined_embeddings = self.multi_head_linear_attention(
                query=item_embeddings,
                key=item_embeddings,
                value=item_embeddings,
                name="item_attention"
            )

            # Residual connection and layer normalization (Eqn. 17)
            refined_embeddings = refined_embeddings * self.residual_weight
            output = tf.contrib.layers.layer_norm(item_embeddings + refined_embeddings)

            return output


class ALPHAModel:
    def __init__(self, num_items, num_users, num_attrs,
                 bipartite_adj, knowledge_adj, laplace_list=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = param.gpu_index
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.num_items = num_items
        self.num_users = num_users
        self.num_attrs = num_attrs
        self.bipartite_adj = bipartite_adj
        self.knowledge_adj = knowledge_adj

        self.batch_size = param.batch_size

        self.ebd_size = param.embedding_size
        self.num_gcn_layers = param.num_gcn_layers
        self.num_attention_heads = param.num_attention_heads
        self.residual_weight = param.residual_weight
        self.is_train = True

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.user, self.item, self.position, self.length, self.target, self.lr, self.dropout_rate = get_inputs()

            with tf.name_scope('ALPHA'):
                # PGCN Module
                self.pgcn = ParallelGraphConvolutionNetwork(
                    num_users=num_users,
                    num_items=num_items,
                    num_attrs=num_attrs,
                    bipartite_adj=bipartite_adj,
                    knowledge_adj=knowledge_adj,
                    embedding_size=self.ebd_size,
                    num_layers=self.num_gcn_layers
                )

                # Get embeddings from PGCN
                self.user_emb, self.item_emb, self.attr_emb = self.pgcn.build()

                # DHAN Module
                self.dhan = DualChannelHybridAttentionNetwork(
                    embedding_size=self.ebd_size,
                    num_heads=self.num_attention_heads,
                    residual_weight=self.residual_weight
                )

                # Apply DHAN
                self.refined_user_emb = self.dhan.traveler_channel(self.user_emb, self.attr_emb)
                self.refined_item_emb = self.dhan.item_channel(self.item_emb)

                # Final prediction
                self.pred = self.final_prediction(self.num_items)

            with tf.name_scope('loss'):
                self.loss_mean = loss_calculation(self.target, self.pred)

            with tf.name_scope('optimizer'):
                self.model_op = optimizer(self.loss_mean, self.lr)

    def final_prediction(self, num_items):
        """Final prediction layer (Eqn. 18)"""
        with tf.variable_scope('prediction'):
            # Get embeddings for input item
            item_embedding = tf.nn.embedding_lookup(self.refined_item_emb, self.item)
            user_embedding = tf.nn.embedding_lookup(self.refined_user_emb, self.user)

            # Aggregate item embeddings (max pooling)
            seq_embedding = tf.reduce_max(item_embedding, 1)

            # Combine with user embedding
            combined_embedding = tf.concat([seq_embedding, user_embedding], axis=1)
            combined_embedding = tf.nn.dropout(combined_embedding, 1 - self.dropout_rate)

            # Final prediction layer
            prediction = tf.layers.dense(
                combined_embedding,
                num_items,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                name='final_prediction'
            )

            return prediction

    def model_training(self, sess, user, item, position, length, target, learning_rate, dropout_rate):
        feed_dict = {self.user: user, self.item: item, self.position: position,
                     self.length: length, self.target: target,
                     self.lr: learning_rate, self.dropout_rate: dropout_rate}
        self.is_train = True

        return sess.run([self.loss_mean, self.model_op], feed_dict)

    def model_evaluation(self, sess, user, item, position, length, target, learning_rate, dropout_rate):
        feed_dict = {self.user: user, self.item: item, self.position: position,
                     self.length: length, self.target: target,
                     self.lr: learning_rate, self.dropout_rate: dropout_rate}
        self.is_train = False

        return sess.run(self.pred, feed_dict)


################## Part 4 模型训练与评估部分 ##################
def train_module(sess, module, batches_train):
    user_all, item_all, position_all, length_all, target_all, train_batch_num = (batches_train[0], batches_train[1],
                                                                                     batches_train[2], batches_train[3],
                                                                                     batches_train[4], batches_train[5])

    shuffled_batch_indexes = np.random.permutation(train_batch_num)
    loss_sum = 0

    for batch_index in shuffled_batch_indexes:
        user = user_all[batch_index]
        item = item_all[batch_index]
        position = position_all[batch_index]
        length = length_all[batch_index]
        target = target_all[batch_index]

        batch_loss, _ = module.model_training(sess=sess, user=user, item=item, position=position, length=length,
                                              target=target,
                                              learning_rate=param.learning_rate, dropout_rate=param.dropout_rate)
        loss_sum += batch_loss

    avg_loss = loss_sum / train_batch_num
    return avg_loss


def evaluate_module(sess, module, batches_test, eval_length):
    user_all, item_all, position_all, length_all, target_all, test_batch_num = (
        batches_test[0], batches_test[1], batches_test[2],
        batches_test[3], batches_test[4], batches_test[5])

    return evaluate_ratings(sess=sess, module=module, user_all=user_all, item_all=item_all,
                            position_all=position_all,
                            length_all=length_all, target_all=target_all, num_batches=test_batch_num,
                            eval_length=eval_length)


def evaluate_ratings(sess, module, user_all, item_all, position_all, length_all, target_all, num_batches,
                     eval_length):
    rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20 = 0, 0, 0, 0, 0, 0

    for batch_index in range(num_batches):
        test_user = user_all[batch_index]
        test_item = item_all[batch_index]
        test_position = position_all[batch_index]
        test_length = length_all[batch_index]
        test_target = target_all[batch_index]

        prediction = module.model_evaluation(sess=sess, user=test_user, item=test_item, position=test_position,
                                             length=test_length, target=test_target, learning_rate=param.learning_rate,
                                             dropout_rate=0)
        recall, mrr = eval_metrics(prediction, test_target, [5, 10, 20])
        rc_5 += recall[0]
        rc_10 += recall[1]
        rc_20 += recall[2]
        mrr_5 += mrr[0]
        mrr_10 += mrr[1]
        mrr_20 += mrr[2]

    return [rc_5 / eval_length, rc_10 / eval_length, rc_20 / eval_length,
            mrr_5 / eval_length, mrr_10 / eval_length, mrr_20 / eval_length]


def eval_metrics(pred_list, target_list, options):
    recall, mrr = [], []
    pred_list = pred_list.argsort()
    for k in options:
        recall.append(0)
        mrr.append(0)
        temp_list = pred_list[:, -k:]
        search_index = 0
        while search_index < len(target_list):
            pos = np.argwhere(temp_list[search_index] == target_list[search_index])
            if len(pos) > 0:
                recall[-1] += 1
                mrr[-1] += 1 / (k - pos[0][0])
            else:
                recall[-1] += 0
                mrr[-1] += 0
            search_index += 1
    return recall, mrr


################## Part 5 打印设置 ##################

def print_train(epoch, loss, time_consumption):
    print('Epoch {} - Training Loss: {:.5f} - Training time: {:.3}'.format(epoch, loss, time_consumption))
    logging.info('Epoch {} - Training Loss: {:.5f} - Training time: {:.3}'.format(epoch, loss, time_consumption))


def print_evaluation(epoch, rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20, test_consumption):
    print("Evaluation at Epoch %d : RC5 = %.4f, RC10 = %.4f, RC20 = %.4f, MRR5 = %.4f, MRR10 = %.4f, MRR20 = %.4f" %
          (epoch, rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20))
    logging.info("Evaluation at Epoch %d : RC5 = %.4f, RC10 = %.4f, RC20 = %.4f, MRR5 = %.4f, MRR10 = %.4f, MRR20 = %.4f"
                 % (epoch, rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20))

    print("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, test_consumption))
    logging.info("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, test_consumption))


param = ParamConfig()
os.environ['CUDA_VISIBLE_DEVICES'] = param.gpu_index

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load dictionaries
    item_dict = load_dict(dict_path=param.item_path)
    user_dict = load_dict(dict_path=param.user_path)
    print("Dictionaries initialized. Loading Data...")

    # Load Data
    train_data = config_input(data_path=param.train_path, item_dict=item_dict, user_dict=user_dict)
    test_data = config_input(data_path=param.test_path, item_dict=item_dict, user_dict=user_dict)

    # Load attribute Data
    user_attr_map, attr_dict = load_attribute_data(
        attr_path=param.attr_path,
        user_dict=user_dict,
        salt=param.salt,
        pseudonymize=param.pseudonymize
    )

    if param.fast_running:
        train_data = train_data[:int(param.fast_ratio * len(train_data))]
        print("Data initialized (Fast Running). Constructing graphs...")
    else:
        print("Data initialized. Constructing graphs...")

    # Construct graphs
    num_users = len(user_dict)
    num_items = len(item_dict)
    num_attrs = len(attr_dict)

    # Construct bipartite graph (traveler-item interactions)
    bipartite_adj = construct_bipartite_graph(train_data, num_users, num_items)

    # Construct knowledge graph (pseudonymized traveler-attribute relationships)
    knowledge_adj, num_attrs_actual = construct_knowledge_graph(
        user_attr_map, attr_dict, num_users, num_items
    )

    print(f"Graphs constructed. Users: {num_users}, Items: {num_items}, Attributes: {num_attrs_actual}")
    print("Generating batches...")

    # Generate batches
    train_batches = generate_batches(input_data=train_data, batch_size=param.batch_size,
                                     padding_num=len(item_dict), is_train=True)
    test_batches = generate_batches(input_data=test_data, batch_size=param.batch_size,
                                    padding_num=len(item_dict), is_train=False)

    print("Batches loaded. Initializing ALPHA network...")

    # Initialize ALPHA model
    module = ALPHAModel(
        num_items=num_items,
        num_users=num_users,
        num_attrs=num_attrs_actual,
        bipartite_adj=bipartite_adj,
        knowledge_adj=knowledge_adj
    )

    print("ALPHA Model Initialized. Start training...")

    # Training loop
    with tf.Session(graph=module.graph, config=module.config) as sess:
        module.sess = sess
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        best_score = -1

        for epoch in range(param.num_epochs):
            time_start = time()
            loss_train = train_module(sess=sess, module=module, batches_train=train_batches)
            time_consumption = time() - time_start

            epoch_num = epoch + 1
            print_train(epoch=epoch_num, loss=loss_train, time_consumption=time_consumption)

            if epoch_num % param.eval_verbose == 0:
                test_start = time()
                [rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20] = \
                    evaluate_module(sess=sess, module=module, batches_test=test_batches,
                                    eval_length=len(test_data))
                test_consumption = time() - test_start

                print_evaluation(epoch_num, rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20, test_consumption)

                if rc_5 >= best_score:
                    best_score = rc_5
                    saver.save(sess, param.check_points, global_step=epoch_num, write_meta_graph=False)
                    print("ALPHA performs better, saving current model....")
                    logging.info("ALPHA performs better, saving current model....")

            # Regenerate training batches for next epoch
            train_batches = generate_batches(input_data=train_data, batch_size=param.batch_size,
                                             padding_num=len(item_dict), is_train=True)

        print("ALPHA training finished.")
        logging.info("ALPHA training finished.")

        print("All process finished.")