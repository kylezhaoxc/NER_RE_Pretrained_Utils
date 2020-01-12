import tensorflow as tf
import os
import json
from tensorflow.python.platform import gfile
from tensorflow.contrib.crf import viterbi_decode

import data_utils
import pickle
import numpy as np

# DEPENDENCY FILES
#1. maps.pkl    -> pickle file containing "char_to_id, id_to_char, tag_to_id, id_to_tag"
#2. input.json  -> dump sample input file.
#3. tmp/tfmodels -> generated model pb file with variables.

# INPUT DATA PROCESS:
#1. Chat Inputs: input sentence CHARS, e.g.[['人','民','邮','电','出','版','社','社','长','王','大','锤']]
#2. Seg Inputs : segmentation features. after Jieba.cut
# - for all bi-chared words: res.extend([1,3])
# - for all uni-chared words: res.extend([0])
# - for all multichard words: res.extend([1,2,2,2,2,2,3])
#3. Targets: [[]]
#4. Dropout: float 1.0



# OUTPUT DATA PROCESS:
# Output I : Project/Reshape:0  shape[1,12,13]  which means[batch_size, num_chars, num_tags]
#            (stand for score)
#
# Output II: crf_loss/transitions:0 shape [14*14] which means [num_tags+2,num_tags+2]
#            (CRF transaction matrix)
# 
# Output III: Sum:0  num_char   
#
#
# To get NER tags, run Viterbi algorithm (decode function in this file)




def decode(logits, lengths, matrix,num_tags):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        with open('maps.pkl', "rb") as f:
            _, _, _, id_to_tag = pickle.load(f)
            tags = [id_to_tag[idx] for idx in paths[0]]
        return tags
    




feed_dict = {}
with open('input.json','r') as rf:
    obj = json.load(rf)
    feed_dict.setdefault('ChatInputs:0',obj[1])
    feed_dict.setdefault('SegInputs:0',obj[2])
    feed_dict.setdefault('Targets:0',obj[3])
    feed_dict.setdefault('Dropout:0',1.0)



# Save ckpt file to pb

# meta_path = 'ckpt\\bilstm\\ner.ckpt.meta' # Your .meta file
# latest_ckp = tf.train.latest_checkpoint('ckpt\\bilstm')
# print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')
# with tf.Session() as save_session:
#     # Restore the graph
#     saver = tf.train.import_meta_graph(meta_path)

#     # Load weights
#     saver.restore(save_session,tf.train.latest_checkpoint('ckpt\\bilstm'))
#     logits_op = save_session.graph.get_tensor_by_name('project/Reshape:0')
#     trans_op = save_session.graph.get_tensor_by_name('crf_loss/transitions:0')
#     len_op = save_session.graph.get_tensor_by_name('Sum:0')
#     chat_op = save_session.graph.get_tensor_by_name('ChatInputs:0')
#     seg_op = save_session.graph.get_tensor_by_name('SegInputs:0')
#     target_op = save_session.graph.get_tensor_by_name('Targets:0')
#     dropout_op = save_session.graph.get_tensor_by_name('Dropout:0')
#        tf.saved_model.simple_save(
#             save_session,
#             'tmp\\tfmodels',
#             inputs={
#                 'ChatInputs:':chat_op,
#                 'SegInputs:':seg_op,
#                 'Targets':target_op,
#                 'Dropout':dropout_op
#                 },
#             outputs={
#                 'project/Reshape':logits_op,
#                 'crf_loss/transitions':trans_op,
#                 'Sum':len_op
#                 })
    



# load simple_saved model and predict with sample data

with tf.Session() as load_session:
    tf.saved_model.loader.load(load_session,[tf.saved_model.tag_constants.SERVING],'tmp\\tfmodels')
    logits_op = load_session.graph.get_tensor_by_name('project/Reshape:0')
    trans_op = load_session.graph.get_tensor_by_name('crf_loss/transitions:0')
    len_op = load_session.graph.get_tensor_by_name('Sum:0')
    score, trans,length = load_session.run([logits_op,trans_op,len_op],feed_dict)
    res = decode(score,length,trans,13)
    print(res)


