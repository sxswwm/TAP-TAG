import json
import nltk
import re
from collections import Counter
import numpy as np
import torch
from gensim.models import word2vec
from nltk.corpus import wordnet
import math
import os
from sklearn.preprocessing import Normalizer


# 获得单词的词性，以便还原为原型
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# 对trigger语句进行处理
def trigger_process(file):
    trigger_processed_file = open('../labeled_rules_processed/trigger_processed_file.txt', mode='w', encoding='utf-8')
    with open(file, 'r') as trigger_file:
        trigger_desc = trigger_file.readline()
        while trigger_desc:
            trigger_desc = json.loads(trigger_desc)['triggerDesc']

            # 去除前面的固定搭配 this trigger fires等
            if 'This Trigger fires ' in trigger_desc:
                trigger_desc = trigger_desc.split('This Trigger fires ', 1)[1]
            elif 'This Trigger ' in trigger_desc:
                trigger_desc = trigger_desc.split('This Trigger ', 1)[1]
            elif 'This Triggers ' in trigger_desc:
                trigger_desc = trigger_desc.split('This Triggers ', 1)[1]
            elif 'Triggers when ' in trigger_desc:
                trigger_desc = trigger_desc.split('Triggers when ', 1)[1]
            elif 'This trigger fires ' in trigger_desc:
                trigger_desc = trigger_desc.split('This trigger fires ', 1)[1]
            if 'will fire ' in trigger_desc:
                trigger_desc = trigger_desc.split('will fire ', 1)[1]
            # 去除后面的注意事项note等
            if 'NOTE:' in trigger_desc:
                trigger_desc = re.sub('NOTE:', '', trigger_desc)
            elif 'Note:' in trigger_desc:
                trigger_desc = re.sub('Note:', '', trigger_desc)  # 去除前面的固定搭配 this triggers
            if 'For example,' in trigger_desc:
                trigger_desc = re.sub('For ', '', trigger_desc)
            # if '/' in trigger_desc:
            #     trigger_desc = re.sub('/', ' or ', trigger_desc)
            # 双空格变成单空格
            if '  ' in trigger_desc:
                trigger_desc = re.sub('  ', ' ', trigger_desc)

            # 去除数字标点非法字符
            remove_chars = '[’!"#$&\'()*+,-.:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]'
            trigger_desc = re.sub(remove_chars, '', trigger_desc)

            # 分词
            trigger_words = [word for word in nltk.word_tokenize(trigger_desc)]

            # 去除固定搭配 every time和 when等
            if trigger_words[0] == 'every' and trigger_words[1] == 'time':
                trigger_words = trigger_words[2:]
            if trigger_words[0] == 'any' and trigger_words[1] == 'time':
                trigger_words = trigger_words[2:]
            if trigger_words[0] == 'when':
                trigger_words = trigger_words[1:]
            if trigger_words[0] == 'whenever':
                trigger_words = trigger_words[1:]
            if trigger_words[0] == 'everytime':
                trigger_words = trigger_words[1:]
            if trigger_words[0] == 'anytime':
                trigger_words = trigger_words[1:]

            # 去除停用词
            # stopwords = nltk.corpus.stopwords.words('english')
            # wordnet停用词太多了，语料太小，去除了所有停用词感觉意思太不通顺，所以自己定义一个停用词序列
            stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                         'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
                         'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                         "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                         'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                         'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                         'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                         'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                         'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                         'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                         'through', 'during', 'before', 'after', 'above', 'below', 'to',
                         'from', 'up', 'down', 'in', 'out', 'over', 'under',
                         'again', 'further', 'then', 'once', 'here', 'there', 'when',
                         'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                         'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                         'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                         'can', 'will', 'just', 'don', "don't", 'should', "should've",
                         'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                         "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
                         "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                         "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
                         'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
                         'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                         'won', "won't", 'wouldn', "wouldn't"]
            trigger_words = [word for word in trigger_words if word not in stopwords]
            # 词形还原
            trigger_words = nltk.pos_tag(trigger_words)
            trigger_words_list = []  # 保存单词原形
            wnl = nltk.WordNetLemmatizer()
            for tag in trigger_words:
                wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                trigger_words_list.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

            trigger_words = trigger_words_list
            trigger_words = [str(i) for i in trigger_words]  # list转化为str
            trigger_desc = " ".join(trigger_words)
            # 应该小写化一下
            trigger_desc = trigger_desc.lower()
            trigger_processed_file.write(trigger_desc + '\n')
            print(trigger_desc)
            trigger_desc = trigger_file.readline()
    trigger_processed_file.close()


# 对action语句进行处理
def action_process(file):
    action_processed_file = open('../labeled_rules_processed/action_processed_file.txt', mode='w', encoding='utf-8')
    with open(file, 'r') as action_file:
        action_desc = action_file.readline()
        while action_desc:
            action_desc = json.loads(action_desc)['actionDesc']

            # 去除前面的固定搭配 this action fires等
            if 'This Action will ' in action_desc:
                action_desc = action_desc.split('This Action will ', 1)[1]
            elif 'This action will ' in action_desc:
                action_desc = action_desc.split('This action will ', 1)[1]
            elif 'This Actions ' in action_desc:
                action_desc = action_desc.split('This Actions ', 1)[1]
            elif 'This Action ' in action_desc:
                action_desc = action_desc.split('This Action ', 1)[1]
            # 去除后面的注意事项note等
            if 'NOTE:' in action_desc:
                action_desc = re.sub('NOTE:', '', action_desc)
            elif 'Note:' in action_desc:
                action_desc = re.sub('Note:', '', action_desc)  # 去除前面的固定搭配 this actions
            if 'For example,' in action_desc:
                action_desc = re.sub('For ', '', action_desc)
            # 双空格变成单空格
            if '  ' in action_desc:
                action_desc = re.sub('  ', ' ', action_desc)
            # if '/' in action_desc:
            #     action_desc = re.sub('/', ' or ', action_desc)

            # 去除数字标点非法字符
            remove_chars = '[’!"#$&\'()*+,-.:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]'
            action_desc = re.sub(remove_chars, '', action_desc)

            # 分词
            action_words = [word for word in nltk.word_tokenize(action_desc)]

            # 去除停用词,wordnet停用词太多了，语料太小，去除了所有停用词感觉意思太不通顺，所以自己定义一个停用词序列
            stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                         'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
                         'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                         "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                         'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                         'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                         'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                         'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                         'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                         'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                         'through', 'during', 'before', 'after', 'above', 'below', 'to',
                         'from', 'up', 'down', 'in', 'out', 'over', 'under',
                         'again', 'further', 'then', 'once', 'here', 'there', 'when',
                         'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                         'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                         'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                         'can', 'will', 'just', 'don', "don't", 'should', "should've",
                         'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                         "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
                         "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                         "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
                         'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
                         'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                         'won', "won't", 'wouldn', "wouldn't"]
            # stopwords = nltk.corpus.stopwords.words('english')

            action_words = [word for word in action_words if word not in stopwords]

            # 词形还原
            action_words = nltk.pos_tag(action_words)
            action_words_list = []  # 保存单词原形
            wnl = nltk.WordNetLemmatizer()
            for tag in action_words:
                wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                action_words_list.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

            action_words = action_words_list
            # list转化为str
            action_words = [str(i) for i in action_words]
            action_desc = " ".join(action_words)
            # 应该小写化一下
            action_desc = action_desc.lower()
            action_processed_file.write(action_desc + '\n')
            print(action_desc)
            action_desc = action_file.readline()
    action_processed_file.close()


# 使用word2vec生成词向量模型
def model_construct(file):
    sentences = word2vec.LineSentence(file)  # 使用linesentence 适合处理一行一句话的文本
    model = word2vec.Word2Vec(sentences, sg=1, window=5, ns_exponent=1, vector_size=128,
                              epochs=140, sample=10e-3, min_count=1)
    return model


# 统计单个单词词频，句子总词数，并将所有句子列表化
def word_counter(processed_data):
    all_words = []
    all_sentences = []
    words_num = 0
    file = open(processed_data, 'r', encoding='utf-8')
    line = file.readline().strip('\n')
    while line:
        all_sentences.append(line)
        words = [word for word in nltk.word_tokenize(line)]
        words_num += len(words)
        for word in words:
            all_words.append(word)
        line = file.readline().strip('\n')
    file.close()
    return Counter(all_words), words_num, all_sentences


# 计算tf-idf,参数为   需要计算的单词，该单词的数量，所有单词的数量，所有的句子
def tf_idf(word, per_word_count, all_words_count, all_sentences):
    sentence_have = 0  # 包含这个单词的句子数
    # 计算tf
    tf = per_word_count / all_words_count
    # 统计含有该单词的句子数
    for i in range(len(all_sentences)):
        if word in all_sentences[i]:
            sentence_have += 1
    # 计算idf,分母加1防止为0
    idf = math.log(len(all_sentences) / (1 + sentence_have))
    # 返回 tf—idf
    return tf * idf


# 计算加权平均句向量,参数为  模型， 每个单词对应的词频， 语料单词总数， 语料中的所有句子组成的列表
def sentence2vec(model, per_word_count, all_words_count, all_sentences, sentence):
    sentence_weight = 0  # 句子总权值
    sentence_vec = np.zeros(128, dtype=float)  # 初始化句向量
    words = [word for word in nltk.word_tokenize(sentence)]
    # # 做一个attention试试
    # words_vec = []
    # for word in words:
    #     words_vec.append(model.wv.__getitem__(word))
    # words_vec = np.array(words_vec)
    # attention = np.matmul(words_vec, words_vec.T)
    # attention = torch.softmax(torch.tensor(attention), dim=1)
    # # softmax后加权求向量
    # new_words_vec = []
    # for att in attention:
    #     word_vec_weighted = torch.sum((att.T.unsqueeze(-1) * torch.tensor(words_vec)), dim=0)
    #     new_words_vec.append(word_vec_weighted)

    for i in range(len(words)):
        word_weight = tf_idf(words[i], per_word_count[words[i]], all_words_count, all_sentences)  # tf-idf计算词权重
        sentence_weight += word_weight
        # 归一化处理
        vec = model.wv.__getitem__(words[i]).reshape(1, -1)
        transformer = Normalizer().fit(vec)
        vec = transformer.transform(vec).squeeze(0)

        sentence_vec += vec * word_weight  # 每个词向量乘以权值加到句向量上
    sentence_vec /= sentence_weight  # 除以一个总权值就是句子的加权平均向量
    return sentence_vec


# 词向量拼接形成句向量
def sentence2vec_cat(model, sentence):
    sentence_vec = np.zeros((1, 128), dtype=float)  # 初始化句向量
    words = [word for word in nltk.word_tokenize(sentence)]
    for i in range(len(words)):
        # 归一化处理
        vec = model.wv.__getitem__(words[i]).reshape(1, -1)
        # transformer = Normalizer().fit(vec)
        # vec = transformer.transform(vec)
        sentence_vec = np.concatenate((sentence_vec, vec), axis=0)

    sentence_vec = sentence_vec[1:]

    return sentence_vec


def trigger_process_per(path, file):
    trigger_processed_file = open('tap_processed_data/trigger_action_processed/' + file, mode='w', encoding='utf-8')
    with open(path + file, 'r', encoding='utf-8') as trigger_file:
        trigger_desc = trigger_file.readline().strip('\n')
        while trigger_desc:
            # 去除前面的固定搭配 this trigger fires等
            if 'This Trigger fires ' in trigger_desc:
                trigger_desc = trigger_desc.split('This Trigger fires ', 1)[1]
            elif 'This Trigger ' in trigger_desc:
                trigger_desc = trigger_desc.split('This Trigger ', 1)[1]
            elif 'This Triggers ' in trigger_desc:
                trigger_desc = trigger_desc.split('This Triggers ', 1)[1]
            elif 'Triggers when ' in trigger_desc:
                trigger_desc = trigger_desc.split('Triggers when ', 1)[1]
            elif 'This trigger fires ' in trigger_desc:
                trigger_desc = trigger_desc.split('This trigger fires ', 1)[1]
            if 'will fire ' in trigger_desc:
                trigger_desc = trigger_desc.split('will fire ', 1)[1]
            # 去除后面的注意事项note等
            if 'NOTE:' in trigger_desc:
                trigger_desc = re.sub('NOTE:', '', trigger_desc)
            elif 'Note:' in trigger_desc:
                trigger_desc = re.sub('Note:', '', trigger_desc)  # 去除前面的固定搭配 this triggers
            if 'For example,' in trigger_desc:
                trigger_desc = re.sub('For ', '', trigger_desc)
            # if '/' in trigger_desc:
            #     trigger_desc = re.sub('/', ' or ', trigger_desc)
            # 双空格变成单空格
            if '  ' in trigger_desc:
                trigger_desc = re.sub('  ', ' ', trigger_desc)

            # 去除数字标点非法字符
            remove_chars = '[’!"#$&\'()*+,-.:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]'
            trigger_desc = re.sub(remove_chars, '', trigger_desc)

            # 分词
            trigger_words = [word for word in nltk.word_tokenize(trigger_desc)]

            # 去除固定搭配 every time和 when等
            if trigger_words[0] == 'every' and trigger_words[1] == 'time':
                trigger_words = trigger_words[2:]
            if trigger_words[0] == 'any' and trigger_words[1] == 'time':
                trigger_words = trigger_words[2:]
            if trigger_words[0] == 'when':
                trigger_words = trigger_words[1:]
            if trigger_words[0] == 'whenever':
                trigger_words = trigger_words[1:]
            if trigger_words[0] == 'everytime':
                trigger_words = trigger_words[1:]
            if trigger_words[0] == 'anytime':
                trigger_words = trigger_words[1:]

            # 去除停用词
            # stopwords = nltk.corpus.stopwords.words('english')
            # wordnet停用词太多了，语料太小，去除了所有停用词感觉意思太不通顺，所以自己定义一个停用词序列
            # stopwords = ['a', 'an', 'you', 'your', 'it', 'is', 'are', 'will', 'has', 'had'
            #                                                                          'that', 'the', 'The', 'be', 'been',
            #              'please', 'to', 'have']
            stopwords = nltk.corpus.stopwords.words('english')

            trigger_words = [word for word in trigger_words if word not in stopwords]

            # 词形还原
            trigger_words = nltk.pos_tag(trigger_words)
            trigger_words_list = []  # 保存单词原形
            wnl = nltk.WordNetLemmatizer()
            for tag in trigger_words:
                wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                trigger_words_list.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

            trigger_words = trigger_words_list
            trigger_words = [str(i) for i in trigger_words]  # list转化为str
            trigger_desc = " ".join(trigger_words)
            # 应该小写化一下
            trigger_desc = trigger_desc.lower()
            trigger_processed_file.write(trigger_desc + '\n')
            print(trigger_desc)
            trigger_desc = trigger_file.readline().strip('\n')
    trigger_processed_file.close()


# 对action语句进行处理
def action_process_per(path, file):
    action_processed_file = open('tap_processed_data/trigger_action_processed/' + file, mode='w', encoding='utf-8')
    with open(path + file, 'r', encoding='utf-8') as action_file:
        action_desc = action_file.readline().strip('\n')
        while action_desc:
            # 去除前面的固定搭配 this action fires等
            if 'This Action will ' in action_desc:
                action_desc = action_desc.split('This Action will ', 1)[1]
            elif 'This action will ' in action_desc:
                action_desc = action_desc.split('This action will ', 1)[1]
            elif 'This Actions ' in action_desc:
                action_desc = action_desc.split('This Actions ', 1)[1]
            elif 'This Action ' in action_desc:
                action_desc = action_desc.split('This Action ', 1)[1]
            # 去除后面的注意事项note等
            if 'NOTE:' in action_desc:
                action_desc = re.sub('NOTE:', '', action_desc)
            elif 'Note:' in action_desc:
                action_desc = re.sub('Note:', '', action_desc)  # 去除前面的固定搭配 this actions
            if 'For example,' in action_desc:
                action_desc = re.sub('For ', '', action_desc)
            # 双空格变成单空格
            if '  ' in action_desc:
                action_desc = re.sub('  ', ' ', action_desc)
            # if '/' in action_desc:
            #     action_desc = re.sub('/', ' or ', action_desc)

            # 去除数字标点非法字符
            remove_chars = '[’!"#$&\'()*+,-.:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]'
            action_desc = re.sub(remove_chars, '', action_desc)

            # 分词
            action_words = [word for word in nltk.word_tokenize(action_desc)]

            # 去除停用词,wordnet停用词太多了，语料太小，去除了所有停用词感觉意思太不通顺，所以自己定义一个停用词序列
            # stopwords = ['a', 'an', 'you', 'your', 'it', 'is', 'are', 'will', 'has', 'had',
            #              'that', 'the', 'The', 'be', 'been', 'please', 'to', 'have']
            stopwords = nltk.corpus.stopwords.words('english')

            action_words = [word for word in action_words if word not in stopwords]

            # 词形还原
            action_words = nltk.pos_tag(action_words)
            action_words_list = []  # 保存单词原形
            wnl = nltk.WordNetLemmatizer()
            for tag in action_words:
                wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                action_words_list.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

            action_words = action_words_list
            # list转化为str
            action_words = [str(i) for i in action_words]
            action_desc = " ".join(action_words)
            # 应该小写化一下
            action_desc = action_desc.lower()
            action_processed_file.write(action_desc + '\n')
            print(action_desc)
            action_desc = action_file.readline().strip('\n')
    action_processed_file.close()


def recipe_process_per(path, file):
    recipe_processed_file = open('../labeled_rules_processed/' + file, mode='w', encoding='utf-8')
    with open(path + file, 'r', encoding='utf-8') as recipe_file:
        recipe_desc = recipe_file.readline().strip('\n')
        while recipe_desc:
            try:
                num, recipe_desc = recipe_desc.split('\t\t')
                if 'This Trigger fires ' in recipe_desc:
                    recipe_desc = recipe_desc.split('This Trigger fires ', 1)[1]
                elif 'This Trigger ' in recipe_desc:
                    recipe_desc = recipe_desc.split('This Trigger ', 1)[1]
                elif 'This Triggers ' in recipe_desc:
                    recipe_desc = recipe_desc.split('This Triggers ', 1)[1]
                elif 'Triggers when ' in recipe_desc:
                    recipe_desc = recipe_desc.split('Triggers when ', 1)[1]
                elif 'This trigger fires ' in recipe_desc:
                    recipe_desc = recipe_desc.split('This trigger fires ', 1)[1]
                elif 'will fire ' in recipe_desc:
                    recipe_desc = recipe_desc.split('will fire ', 1)[1]
                # 去除前面的固定搭配 this action fires等
                elif 'This Action will ' in recipe_desc:
                    recipe_desc = recipe_desc.split('This Action will ', 1)[1]
                elif 'This action will ' in recipe_desc:
                    recipe_desc = recipe_desc.split('This action will ', 1)[1]
                elif 'This Actions ' in recipe_desc:
                    recipe_desc = recipe_desc.split('This Actions ', 1)[1]
                elif 'This Action ' in recipe_desc:
                    recipe_desc = recipe_desc.split('This Action ', 1)[1]
                # 去除后面的注意事项note等
                if 'NOTE:' in recipe_desc:
                    recipe_desc = re.sub('NOTE:', '', recipe_desc)
                elif 'Note:' in recipe_desc:
                    recipe_desc = re.sub('Note:', '', recipe_desc)  # 去除前面的固定搭配 this actions
                if 'For example,' in recipe_desc:
                    recipe_desc = re.sub('For ', '', recipe_desc)
                # 双空格变成单空格
                if '  ' in recipe_desc:
                    recipe_desc = re.sub('  ', ' ', recipe_desc)
                # if '/' in recipe_desc:
                #     recipe_desc = re.sub('/', ' or ', recipe_desc)

                # 去除数字标点非法字符
                remove_chars = '[’!"#$&\'()*+,-.:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]'
                recipe_desc = re.sub(remove_chars, '', recipe_desc)

                # 分词
                recipe_words = [word for word in nltk.word_tokenize(recipe_desc)]
                # 去除固定搭配 every time和 when等
                if recipe_words[0] == 'every' and recipe_words[1] == 'time':
                    recipe_words = recipe_words[2:]
                if recipe_words[0] == 'any' and recipe_words[1] == 'time':
                    recipe_words = recipe_words[2:]
                if recipe_words[0] == 'when':
                    recipe_words = recipe_words[1:]
                if recipe_words[0] == 'whenever':
                    recipe_words = recipe_words[1:]
                if recipe_words[0] == 'everytime':
                    recipe_words = recipe_words[1:]
                if recipe_words[0] == 'anytime':
                    recipe_words = recipe_words[1:]

                # 去除停用词,wordnet停用词太多了，语料太小，去除了所有停用词感觉意思太不通顺，所以自己定义一个停用词序列
                # stopwords = ['a', 'an', 'you', 'your', 'it', 'is', 'are', 'will', 'has', 'had',
                #              'that', 'the', 'The', 'be', 'been', 'please', 'to', 'have']
                # stopwords = nltk.corpus.stopwords.words('english')
                #
                stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                             'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
                             'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                             "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                             'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                             'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                             'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                             'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                             'through', 'during', 'before', 'after', 'above', 'below', 'to',
                             'from', 'up', 'down', 'in', 'out', 'over', 'under',
                             'again', 'further', 'then', 'once', 'here', 'there', 'when',
                             'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                             'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                             'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                             'can', 'will', 'just', 'don', "don't", 'should', "should've",
                             'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                             "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
                             "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                             "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
                             'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
                             'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                             'won', "won't", 'wouldn', "wouldn't"]
                recipe_words = [word for word in recipe_words if word not in stopwords]

                # 词形还原
                recipe_words = nltk.pos_tag(recipe_words)
                recipe_words_list = []  # 保存单词原形
                wnl = nltk.WordNetLemmatizer()
                for tag in recipe_words:
                    wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                    recipe_words_list.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

                recipe_words = recipe_words_list
                # list转化为str
                recipe_words = [str(i) for i in recipe_words]
                recipe_desc = " ".join(recipe_words)
                # 应该小写化一下
                recipe_desc = recipe_desc.lower()
                recipe_processed_file.write(num + '\t\t' + recipe_desc + '\n')
                print(recipe_desc)
                recipe_desc = recipe_file.readline().strip('\n')
            except:
                continue
    recipe_processed_file.close()

if __name__ == '__main__':
    # 对trigger和action预处理，保存在trigger_processed_file 和 action_processed_file
    # 将训练后的词向量模型存入save中
    # trigger_process('../ifttt_dataset/triggerList.json')
    # action_process('../ifttt_dataset/actionList.json')
    # trigger_model = model_construct('tap_processed_data/trigger_processed_file.txt')
    # trigger_model.save('save/trigger.model')
    # action_model = model_construct('tap_processed_data/action_processed_file.txt')
    # action_model.save('save/action.model')
    # recipe_model = model_construct('../recipes.txt')
    # recipe_model.save('../save/recipe.model')

    # trigger_model = word2vec.Word2Vec.load('../save/recipe.model')
    # sims = trigger_model.wv.most_similar('car', topn=10)
    # print(sims)
    # t_per_word_count, t_all_words_count, t_all_sentences = word_counter('tap_processed_data/trigger_processed_file.txt')
    # triggers_vectors = sentence2vec(trigger_model, t_per_word_count, t_all_words_count, t_all_sentences)
    # a_per_word_count, a_all_words_count, a_all_sentences = word_counter('tap_processed_data/action_processed_file.txt')
    # actions_vectors = sentence2vec(action_model, a_per_word_count, a_all_words_count, a_all_sentences)
    # trigger_model = word2vec.Word2Vec.load('save/trigger.model')
    # action_model = word2vec.Word2Vec.load('save/action.model')

    files = os.listdir('../labeled_rules/')
    for file in files:
        recipe_process_per('../labeled_rules/', file)

