import tensorflow as tf
import os
import json

import pickle
import numpy as np




id2relation = {}
f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
while True:
    content = f.readline()
    if content == '':
        break
    content = content.strip().split()
    id2relation[int(content[1])] = content[0]
f.close()

ent1 = np.load('Sample_Input\\Ent1.npy')
ent2 = np.load('Sample_Input\\Ent2.npy')
input_word = np.load('Sample_Input\\Input_Word.npy')
rel = np.load('Sample_Input\\Rel.npy')
shape = np.load('Sample_Input\\Total_Shape.npy')


# Sample Input Word:
# 实体1: 李晓华
# 实体2: 王大牛
# 王大牛命令李晓华在周末前完成这份代码。
# Model Input:
# - total_shape:(2,)
# - input_word: (1,70)
# - input_pos1: (1,70) see position embedding 
# - input_pos2: (1,70) same as above, ref testGRU.py Line 247~252
# - input_y: (1,tag_num) = (1,12) [1,0,0,0,0.......]

# Ref : testGRU Line 141 

# Model Output :
# - softmax probs for all categories

feed_dict = {
    'model/total_shape:0': shape,
    'model/input_word:0':input_word,
    'model/input_pos1:0':ent1,
    'model/input_pos2:0':ent2,
    'model/input_y:0':rel
}
with tf.Session() as load_session:
    tf.saved_model.loader.load(load_session,[tf.saved_model.tag_constants.SERVING],'tempmodel')
    softmax_op = load_session.graph.get_tensor_by_name('model/Softmax_2:0')
    prob = load_session.run(softmax_op,feed_dict)
    print("关系是:")
    #print(prob)
    top3_id = prob.argsort()[-3:][::-1]
    for n, rel_id in enumerate(top3_id):
        print("No." + str(n+1) + ": " + id2relation[rel_id] + ", Probability is " + str(prob[rel_id]))