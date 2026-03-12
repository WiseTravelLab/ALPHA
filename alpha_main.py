# ALPHA: Attribute-augmented Lightweight Privacy-preserving Hybrid Attentive network for Tourism Recommendation
# Tensorflow v1 implementation (tensorflow-gpu=1.13.0)
# Updated version: Pre-training + Prompt-tuning with Linear Cross-Attention

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


################## Part 1 Hyperparam ##################
class ParamConfig:
    def __init__(self):
        '''
           block 1: the hyper parameters for model training
        '''
        self.learning_rate = 0.001
        self.dropout_rate = 0.1
        self.batch_size = 256
        self.num_epochs = 60
        self.pretrain_epochs = 30
        self.eval_verbose = 10
        self.fast_running = False
        self.fast_ratio = 0.5

        '''
            block 2: the hyper parameters for ALPHA model
        '''
        self.embedding_size = 64
        self.num_layers = 2
        self.num_attention_heads = 2
        self.residual_weight = 1.0
        self.gpu_index = '0'

        '''
            block 3: dataset and privacy settings
        '''
        self.dataset = "TripAdvisor-Hotel"
        self.train_path = "../Data/" + self.dataset + "/train_data.txt"
        self.test_path = "../Data/" + self.dataset + "/test_data.txt"
        self.attr_path = "../Data/" + self.dataset + "/attr_data.txt"
        self.item_path = "../Data/" + self.dataset + "/item_dict.txt"
        self.user_path = "../Data/" + self.dataset + "/user_dict.txt"
        self.check_points = "../check_points/" + self.dataset + ".ckpt"
        self.pretrain_check_points = "../check_points/" + self.dataset + "_pretrain.ckpt"

        self.salt = b"ALPHA_PRIVACY_SALT_2025"
        self.pseudonymize = True


################## Part 2 Dataloader ##################
def load_dict(dict_path):
    dict_output = {}
    with open(dict_path, 'r') as file_object:
        elements = file_object.readlines()
    for dict_element in elements:
        dict_element = dict_element.strip().split('\t')
        dict_output[dict_element[1]] = int(dict_element[0])
    return dict_output


def pseudonymize_user(user_id, salt):
    hash_input = str(user_id).encode() + salt
    hashed = hashlib.sha256(hash_input).hexdigest()
    return int(hashed[:8], 16) % (2 ** 31 - 1)


def load_attribute_data(attr_path, user_dict, salt, pseudonymize=True):
    with open(attr_path, 'r') as f:
        lines = f.readlines()

    user_attr_map = {}
    attr_set = set()
    pseudonymize_vector = []

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue

        user_id = parts[0]
        if user_id not in user_dict:
            continue

        user_idx = user_dict[user_id]
        should_pseudonymize = pseudonymize and len(parts[1].strip()) > 0
        
        if should_pseudonymize:
            user_idx = pseudonymize_user(user_idx, salt)
            pseudonymize_vector.append(1)
        else:
            pseudonymize_vector.append(0)

        attributes = parts[1].split(',')
        user_attr_map[user_idx] = []

        for attr in attributes:
            if attr:
                attr_set.add(attr)
                user_attr_map[user_idx].append(attr)

    attr_dict = {attr: idx for idx, attr in enumerate(sorted(list(attr_set)))}
    return user_attr_map, attr_dict, pseudonymize_vector


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
            temp.append(item)
            temp.append(position)
            temp.append(length)
            temp.append(item_dict[line_split[-1]])
            data.append(temp)
    return data


def construct_bipartite_graph(data, num_users, num_items):
    user_item_pairs = []

    for record in data:
        item = record[0]
        user = int(item[0])
        items = [int(i) for i in item[1:]]

        for item in items:
            user_item_pairs.append([user, item])

    rows = [pair[0] for pair in user_item_pairs]
    cols = [pair[1] + num_users for pair in user_item_pairs]
    values = [1.0] * len(rows)

    rows_rev = cols
    cols_rev = rows
    values_rev = values

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
            user_nodes.append(user_idx)
            attr_nodes.append(num_users + num_items + attr_idx)
            values.append(1.0)
            attr_nodes.append(num_users + num_items + attr_idx)
            user_nodes.append(user_idx)
            values.append(1.0)

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
    num_batches = int(len(input_data) / batch_size)

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


################## Part 3 ALPHA model ##################

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
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
    return user, item, position, length, target, learning_rate, dropout_rate, is_training


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


class ProfileAwarePrivacyChannel:
    def __init__(self, num_users, num_items, num_attrs,
                 knowledge_adj, embedding_size, num_layers):
        self.num_users = num_users
        self.num_items = num_items
        self.num_attrs = num_attrs
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.total_nodes = num_users + num_items + num_attrs
        self.knowledge_adj_norm = normalize_adjacency_matrix(knowledge_adj)

    def build(self, scope="profile_channel"):
        with tf.variable_scope(scope):
            traveler_embeddings = tf.get_variable(
                'traveler_embeddings',
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

            entity_embeddings = tf.concat([traveler_embeddings, item_embeddings, attr_embeddings], axis=0)
            adj_tensor = _convert_sp_mat_to_sp_tensor(self.knowledge_adj_norm)

            all_layer_embeddings = [entity_embeddings]
            current_embeddings = entity_embeddings

            for layer in range(self.num_layers):
                current_embeddings = tf.sparse_tensor_dense_matmul(adj_tensor, current_embeddings)
                
                W = tf.get_variable(f'W_gcn_{layer}', 
                                   shape=[self.embedding_size, self.embedding_size],
                                   initializer=tf.contrib.layers.xavier_initializer())
                current_embeddings = tf.matmul(current_embeddings, W)
                current_embeddings = tf.nn.relu(current_embeddings)
                
                all_layer_embeddings.append(current_embeddings)

            stacked = tf.stack(all_layer_embeddings, axis=1)
            final_embeddings = tf.reduce_mean(stacked, axis=1)

            traveler_final, item_final, attr_final = tf.split(
                final_embeddings, 
                [self.num_users, self.num_items, self.num_attrs], 
                axis=0
            )

            return traveler_final, item_final, attr_final


class GraphTransformerLayer:
    def __init__(self, embedding_size, num_heads, dropout_rate):
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.dropout_rate = dropout_rate

    def multi_head_attention(self, query, key, value, scope="mha"):
        with tf.variable_scope(scope):
            Qs = []
            Ks = []
            Vs = []
            
            for h in range(self.num_heads):
                W_q = tf.get_variable(f'W_q_{h}', 
                                     shape=[self.embedding_size, self.head_dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
                W_k = tf.get_variable(f'W_k_{h}',
                                     shape=[self.embedding_size, self.head_dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
                W_v = tf.get_variable(f'W_v_{h}',
                                     shape=[self.embedding_size, self.head_dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
                
                Qs.append(tf.matmul(query, W_q))
                Ks.append(tf.matmul(key, W_k))
                Vs.append(tf.matmul(value, W_v))
            
            Q = tf.concat(Qs, axis=-1)
            K = tf.concat(Ks, axis=-1)
            V = tf.concat(Vs, axis=-1)
            
            scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(self.embedding_size, tf.float32))
            attention_weights = tf.nn.softmax(scores)
            attention_weights = tf.nn.dropout(attention_weights, 1 - self.dropout_rate)
            
            output = tf.matmul(attention_weights, V)
            
            W_o = tf.get_variable('W_o', 
                                 shape=[self.embedding_size, self.embedding_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            output = tf.matmul(output, W_o)
            
            return output

    def feed_forward(self, inputs, scope="ffn"):
        with tf.variable_scope(scope):
            hidden = tf.layers.dense(inputs, self.embedding_size * 4, activation=tf.nn.gelu)
            hidden = tf.nn.dropout(hidden, 1 - self.dropout_rate)
            output = tf.layers.dense(hidden, self.embedding_size)
            return output

    def __call__(self, traveler_emb, item_emb, scope="gt_layer"):
        with tf.variable_scope(scope):
            traveler_attended = self.multi_head_attention(
                traveler_emb, item_emb, item_emb, "traveler_attention"
            )
            traveler_attended = tf.nn.dropout(traveler_attended, 1 - self.dropout_rate)
            traveler_emb = tf.contrib.layers.layer_norm(traveler_emb + traveler_attended)
            
            item_attended = self.multi_head_attention(
                item_emb, traveler_emb, traveler_emb, "item_attention"
            )
            item_attended = tf.nn.dropout(item_attended, 1 - self.dropout_rate)
            item_emb = tf.contrib.layers.layer_norm(item_emb + item_attended)
            
            traveler_ffn = self.feed_forward(traveler_emb, "traveler_ffn")
            traveler_ffn = tf.nn.dropout(traveler_ffn, 1 - self.dropout_rate)
            traveler_emb = tf.contrib.layers.layer_norm(traveler_emb + traveler_ffn)
            
            item_ffn = self.feed_forward(item_emb, "item_ffn")
            item_ffn = tf.nn.dropout(item_ffn, 1 - self.dropout_rate)
            item_emb = tf.contrib.layers.layer_norm(item_emb + item_ffn)
            
            return traveler_emb, item_emb


class LinearCrossAttention:
    def __init__(self, embedding_size, num_heads, residual_weight, dropout_rate):
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.residual_weight = residual_weight
        self.dropout_rate = dropout_rate

    def compute_attribute_map(self, attribute_prompts, scope="attr_map"):
        with tf.variable_scope(scope):
            head_maps = []
            
            for h in range(self.num_heads):
                W_k = tf.get_variable(f'W_K_{h}', 
                                     shape=[self.embedding_size, self.head_dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
                W_v = tf.get_variable(f'W_V_{h}',
                                     shape=[self.embedding_size, self.head_dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
                
                K_h = tf.matmul(attribute_prompts, W_k)
                V_h = tf.matmul(attribute_prompts, W_v)
                
                kt_v_h = tf.matmul(K_h, V_h, transpose_a=True)
                kt_v_h = kt_v_h / tf.sqrt(tf.cast(self.head_dim, tf.float32))
                
                head_maps.append(kt_v_h)
            
            attr_map = tf.stack(head_maps, axis=0)
            return attr_map

    def refine_with_map(self, queries, attr_map, is_item=True, scope="refine"):
        with tf.variable_scope(scope):
            refined_heads = []
            
            for h in range(self.num_heads):
                W_q = tf.get_variable(f'W_Q_{h}',
                                     shape=[self.embedding_size, self.head_dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
                
                Q_h = tf.matmul(queries, W_q)
                map_h = attr_map[h]
                refined_h = tf.matmul(Q_h, map_h)
                refined_heads.append(refined_h)
            
            refined = tf.concat(refined_heads, axis=1)
            
            W_o = tf.get_variable('W_o', 
                                 shape=[self.embedding_size, self.embedding_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            refined = tf.matmul(refined, W_o)
            
            return refined

    def __call__(self, pretrained_emb, attribute_prompts, is_item=True):
        attr_map = self.compute_attribute_map(attribute_prompts)
        refined = self.refine_with_map(pretrained_emb, attr_map, is_item)
        refined = refined * self.residual_weight
        output = tf.contrib.layers.layer_norm(pretrained_emb + refined)
        return output


class ALPHAModel:
    def __init__(self, num_items, num_users, num_attrs,
                 bipartite_adj, knowledge_adj):
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
        self.num_layers = param.num_layers
        self.num_attention_heads = param.num_attention_heads
        self.residual_weight = param.residual_weight

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.user, self.item, self.position, self.length, self.target, self.lr, self.dropout_rate, self.is_training = get_inputs()

            with tf.name_scope('ALPHA'):
                self.profile_channel = ProfileAwarePrivacyChannel(
                    num_users=num_users,
                    num_items=num_items,
                    num_attrs=num_attrs,
                    knowledge_adj=knowledge_adj,
                    embedding_size=self.ebd_size,
                    num_layers=self.num_layers
                )
                
                self.profile_traveler_emb, self.profile_item_emb, self.attr_emb = self.profile_channel.build()
                self.attribute_prompts = self.attr_emb

                self.gt_layers = []
                for i in range(self.num_layers):
                    gt_layer = GraphTransformerLayer(
                        embedding_size=self.ebd_size,
                        num_heads=self.num_attention_heads,
                        dropout_rate=self.dropout_rate
                    )
                    self.gt_layers.append(gt_layer)

                self.pretrained_traveler_emb = tf.get_variable(
                    'pretrained_traveler_emb',
                    shape=[num_users, self.ebd_size],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    trainable=True
                )
                self.pretrained_item_emb = tf.get_variable(
                    'pretrained_item_emb',
                    shape=[num_items, self.ebd_size],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    trainable=True
                )

                self.linear_cross_attn = LinearCrossAttention(
                    embedding_size=self.ebd_size,
                    num_heads=self.num_attention_heads,
                    residual_weight=self.residual_weight,
                    dropout_rate=self.dropout_rate
                )

                self.refined_item_emb = self.linear_cross_attn(
                    self.pretrained_item_emb, 
                    self.attribute_prompts,
                    is_item=True
                )
                self.refined_traveler_emb = self.linear_cross_attn(
                    self.pretrained_traveler_emb,
                    self.attribute_prompts,
                    is_item=False
                )

                self.pred = self.final_prediction()

            with tf.name_scope('loss'):
                self.loss_mean = loss_calculation(self.target, self.pred)
                self.pretrain_loss = self.compute_pretrain_loss()

            with tf.name_scope('optimizer'):
                self.model_op = optimizer(self.loss_mean, self.lr)
                self.pretrain_op = optimizer(self.pretrain_loss, self.lr)

    def compute_pretrain_loss(self):
        traveler_emb = self.pretrained_traveler_emb
        item_emb = self.pretrained_item_emb
        
        for i, gt_layer in enumerate(self.gt_layers):
            traveler_emb, item_emb = gt_layer(
                traveler_emb, item_emb, 
                scope=f"gt_layer_{i}"
            )
        
        user_emb_seq = tf.nn.embedding_lookup(traveler_emb, self.user)
        item_emb_seq = tf.nn.embedding_lookup(item_emb, self.item)
        
        seq_embedding = tf.reduce_max(item_emb_seq, 1)
        
        scores = tf.matmul(seq_embedding, item_emb, transpose_b=True)
        
        pretrain_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.target, logits=scores
        )
        pretrain_loss = tf.reduce_mean(pretrain_loss)
        
        return pretrain_loss

    def final_prediction(self):
        with tf.variable_scope('prediction'):
            item_embedding = tf.nn.embedding_lookup(self.refined_item_emb, self.item)
            user_embedding = tf.nn.embedding_lookup(self.refined_traveler_emb, self.user)

            seq_embedding = tf.reduce_max(item_embedding, 1)

            combined_embedding = tf.concat([seq_embedding, user_embedding], axis=1)
            combined_embedding = tf.nn.dropout(combined_embedding, 1 - self.dropout_rate)

            W_f = tf.get_variable('W_f', 
                                 shape=[self.ebd_size * 2, self.num_items],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b_f = tf.get_variable('b_f', 
                                 shape=[self.num_items],
                                 initializer=tf.zeros_initializer())
            
            prediction = tf.matmul(combined_embedding, W_f) + b_f

            return prediction

    def pretrain_step(self, sess, user, item, position, length, target, learning_rate, dropout_rate):
        feed_dict = {
            self.user: user, 
            self.item: item, 
            self.position: position,
            self.length: length, 
            self.target: target,
            self.lr: learning_rate, 
            self.dropout_rate: dropout_rate,
            self.is_training: True
        }
        return sess.run([self.pretrain_loss, self.pretrain_op], feed_dict)

    def finetune_step(self, sess, user, item, position, length, target, learning_rate, dropout_rate):
        feed_dict = {
            self.user: user, 
            self.item: item, 
            self.position: position,
            self.length: length, 
            self.target: target,
            self.lr: learning_rate, 
            self.dropout_rate: dropout_rate,
            self.is_training: True
        }
        return sess.run([self.loss_mean, self.model_op], feed_dict)

    def evaluate(self, sess, user, item, position, length, target, learning_rate, dropout_rate):
        feed_dict = {
            self.user: user, 
            self.item: item, 
            self.position: position,
            self.length: length, 
            self.target: target,
            self.lr: learning_rate, 
            self.dropout_rate: dropout_rate,
            self.is_training: False
        }
        return sess.run(self.pred, feed_dict)


################## Part 4 model training and evaluations ##################
def pretrain_module(sess, module, batches_train):
    user_all, item_all, position_all, length_all, target_all, train_batch_num = batches_train

    shuffled_batch_indexes = np.random.permutation(train_batch_num)
    loss_sum = 0

    for batch_index in shuffled_batch_indexes:
        user = user_all[batch_index]
        item = item_all[batch_index]
        position = position_all[batch_index]
        length = length_all[batch_index]
        target = target_all[batch_index]

        batch_loss, _ = module.pretrain_step(
            sess=sess, user=user, item=item, position=position, length=length,
            target=target, learning_rate=param.learning_rate, dropout_rate=param.dropout_rate
        )
        loss_sum += batch_loss

    avg_loss = loss_sum / train_batch_num
    return avg_loss


def finetune_module(sess, module, batches_train):
    user_all, item_all, position_all, length_all, target_all, train_batch_num = batches_train

    shuffled_batch_indexes = np.random.permutation(train_batch_num)
    loss_sum = 0

    for batch_index in shuffled_batch_indexes:
        user = user_all[batch_index]
        item = item_all[batch_index]
        position = position_all[batch_index]
        length = length_all[batch_index]
        target = target_all[batch_index]

        batch_loss, _ = module.finetune_step(
            sess=sess, user=user, item=item, position=position, length=length,
            target=target, learning_rate=param.learning_rate, dropout_rate=param.dropout_rate
        )
        loss_sum += batch_loss

    avg_loss = loss_sum / train_batch_num
    return avg_loss


def evaluate_module(sess, module, batches_test, eval_length):
    user_all, item_all, position_all, length_all, target_all, test_batch_num = batches_test

    return evaluate_ratings(
        sess=sess, module=module, user_all=user_all, item_all=item_all,
        position_all=position_all, length_all=length_all, target_all=target_all, 
        num_batches=test_batch_num, eval_length=eval_length
    )


def evaluate_ratings(sess, module, user_all, item_all, position_all, length_all, target_all, num_batches,
                     eval_length):
    rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20 = 0, 0, 0, 0, 0, 0

    for batch_index in range(num_batches):
        test_user = user_all[batch_index]
        test_item = item_all[batch_index]
        test_position = position_all[batch_index]
        test_length = length_all[batch_index]
        test_target = target_all[batch_index]

        prediction = module.evaluate(
            sess=sess, user=test_user, item=test_item, position=test_position,
            length=test_length, target=test_target, 
            learning_rate=param.learning_rate, dropout_rate=0
        )
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


################## Part 5 printer ##################

def print_train(epoch, loss, time_consumption, phase="Training"):
    print('Epoch {} - {} Loss: {:.5f} - Time: {:.3f}s'.format(epoch, phase, loss, time_consumption))
    logging.info('Epoch {} - {} Loss: {:.5f} - Time: {:.3f}s'.format(epoch, phase, loss, time_consumption))


def print_evaluation(epoch, rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20, test_consumption):
    print("Evaluation at Epoch %d : RC5 = %.4f, RC10 = %.4f, RC20 = %.4f, MRR5 = %.4f, MRR10 = %.4f, MRR20 = %.4f" %
          (epoch, rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20))
    logging.info("Evaluation at Epoch %d : RC5 = %.4f, RC10 = %.4f, RC20 = %.4f, MRR5 = %.4f, MRR10 = %.4f, MRR20 = %.4f"
                 % (epoch, rc_5, rc_10, rc_20, mrr_5, mrr_10, mrr_20))

    print("Epoch: {}, Recommender evaluate time: {:.3f}s".format(epoch, test_consumption))
    logging.info("Epoch: {}, Recommender evaluate time: {:.3f}s".format(epoch, test_consumption))


param = ParamConfig()
os.environ['CUDA_VISIBLE_DEVICES'] = param.gpu_index

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    item_dict = load_dict(dict_path=param.item_path)
    user_dict = load_dict(dict_path=param.user_path)
    print("Dictionaries initialized. Loading Data...")

    train_data = config_input(data_path=param.train_path, item_dict=item_dict, user_dict=user_dict)
    test_data = config_input(data_path=param.test_path, item_dict=item_dict, user_dict=user_dict)

    user_attr_map, attr_dict, pseudonymize_vector = load_attribute_data(
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

    num_users = len(user_dict)
    num_items = len(item_dict)
    num_attrs = len(attr_dict)

    bipartite_adj = construct_bipartite_graph(train_data, num_users, num_items)
    knowledge_adj, num_attrs_actual = construct_knowledge_graph(
        user_attr_map, attr_dict, num_users, num_items
    )

    print(f"Graphs constructed. Users: {num_users}, Items: {num_items}, Attributes: {num_attrs_actual}")
    print("Generating batches...")

    train_batches = generate_batches(input_data=train_data, batch_size=param.batch_size,
                                     padding_num=len(item_dict), is_train=True)
    test_batches = generate_batches(input_data=test_data, batch_size=param.batch_size,
                                    padding_num=len(item_dict), is_train=False)

    print("Batches loaded. Initializing ALPHA network...")

    module = ALPHAModel(
        num_items=num_items,
        num_users=num_users,
        num_attrs=num_attrs_actual,
        bipartite_adj=bipartite_adj,
        knowledge_adj=knowledge_adj
    )

    print("ALPHA Model Initialized. Start pre-training...")

    with tf.Session(graph=module.graph, config=module.config) as sess:
        module.sess = sess
        saver = tf.train.Saver()
        pretrain_saver = tf.train.Saver(var_list=[module.pretrained_traveler_emb, module.pretrained_item_emb])
        sess.run(tf.global_variables_initializer())

        best_pretrain_loss = float('inf')
        for epoch in range(param.pretrain_epochs):
            time_start = time()
            loss_pretrain = pretrain_module(sess=sess, module=module, batches_train=train_batches)
            time_consumption = time() - time_start

            epoch_num = epoch + 1
            print_train(epoch=epoch_num, loss=loss_pretrain, time_consumption=time_consumption, phase="Pre-training")

            if loss_pretrain < best_pretrain_loss:
                best_pretrain_loss = loss_pretrain
                pretrain_saver.save(sess, param.pretrain_check_points, global_step=epoch_num, write_meta_graph=False)
                print("Pre-training loss improved, saving current model....")

            train_batches = generate_batches(input_data=train_data, batch_size=param.batch_size,
                                             padding_num=len(item_dict), is_train=True)

        print("Pre-training finished. Starting fine-tuning...")

        best_score = -1
        for epoch in range(param.num_epochs):
            time_start = time()
            loss_train = finetune_module(sess=sess, module=module, batches_train=train_batches)
            time_consumption = time() - time_start

            epoch_num = epoch + 1
            print_train(epoch=epoch_num, loss=loss_train, time_consumption=time_consumption, phase="Fine-tuning")

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

            train_batches = generate_batches(input_data=train_data, batch_size=param.batch_size,
                                             padding_num=len(item_dict), is_train=True)

        print("ALPHA training finished.")
        logging.info("ALPHA training finished.")

        print("All process finished.")
