#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate  # 导入这个函数
from torch.utils.data import Dataset
from word2vec import word_counter, sentence2vec, sentence2vec_cat
from gensim.models import word2vec

ENTITY_LABEL_NAMES = [
    'air_conditioner', 'Android', 'calendar', 'camera', 'car', 'dog', 'timer', 'book',
    'door', 'dryer', 'email', 'Facebook', 'light', 'oven', 'rain', 'window', 'photo',
    'refrigerator', 'switch', 'telephone', 'thermostat', 'Google', 'humidity',
    'message', 'video']
# 'Developer tools', 'News & information', 'Email', 'Business tools', 'Appliances', 'Mobile devices & accessories', 'Lighting', 'Music', 'Smart hubs & systems', 'Connected car', 'Notes', 'Power monitoring & management', 'Journaling & personal data', 'Gardening', 'Clocks & displays', 'Cloud storage', 'DIY electronics', 'Blogging', 'Routers & computer accessories', 'Machine learning', 'Finance & payments', 'Restaurants & food', 'Tags & beacons', 'Non-profits', 'Social networks', 'Contacts', 'Environment control & monitoring', 'Time management & tracking', 'Security & monitoring systems', 'Shopping', 'Blinds', 'Health & fitness', 'Government', 'Calendars & scheduling', 'Communication', 'Bookmarking', 'Television & cable', 'Survey tools', 'Weather', 'Task management & to-dos', 'Pet trackers', 'Notifications', 'Travel & transit', 'Photo & video', 'Voice assistants']


def get_all_labeled_recipes(recipe_model):
    rules_ebds = list()
    labels = list()
    files = os.listdir('labeled_rules_processed/')
    for file in files:
        if 'test' not in file:
            label = file.split('.txt')[0]
            file = open('labeled_rules_processed/' + file, mode='r')
            for line in file:
                rule = line.strip('\n').split('\t\t')[1]
                rule_ebd = sentence2vec_cat(recipe_model, rule)
                rule_ebd = torch.tensor(rule_ebd).unsqueeze(0)
                rules_ebds.append(rule_ebd)
                labels.append(ENTITY_LABEL_NAMES.index(label))

            file.close()
        else:
            continue

    return rules_ebds, labels


class TAPTrainDataset(Dataset):
    def __init__(self, rules_dir, tap_negative_sample_size):
        self.rules_dir = rules_dir
        self.len = len(ENTITY_LABEL_NAMES)
        self.entity_labels = ENTITY_LABEL_NAMES
        self.negative_sample_size = tap_negative_sample_size
        self.per_word_count, self.all_words_count, self.all_sentences = word_counter('recipes.txt')
        self.recipe_model = word2vec.Word2Vec.load(
            'save/recipe.model'
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_label = self.entity_labels[idx]

        positive_nums = list()
        positive_rules = list()
        positive_rules_ebds = list()
        negative_nums = list()
        negative_rules = list()
        negtive_rules_ebds = list()

        positive_file = open(self.rules_dir + positive_label + '.txt', mode='r')

        for positive_line in positive_file:
            positve_num, positive_rule = positive_line.split('\t\t')
            positive_rule_ebd = sentence2vec_cat(self.recipe_model, positive_rule)

            positive_nums.append(int(positve_num))
            positive_rules.append(positive_rule)
            positive_rules_ebds.append(positive_rule_ebd)

        recipe_file_len = len(self.all_sentences)

        negative_sample_list = list()
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(recipe_file_len, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                positive_nums,
                assume_unique=True,
                invert=True
            )

            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        for num in negative_sample:
            negative_num, negtive_rule = self.all_sentences[num].split('\t\t')
            negative_rule_ebd = sentence2vec_cat(self.recipe_model, negtive_rule)
            negative_nums.append(negative_num)
            negative_rules.append(negtive_rule)
            negtive_rules_ebds.append(negative_rule_ebd)

        positive_file.close()
        return positive_nums, positive_rules_ebds, negtive_rules_ebds


class TAPTestDataset(Dataset):
    def __init__(self, rules_dir):
        self.rules_dir = rules_dir
        self.len = len(ENTITY_LABEL_NAMES)
        self.entity_labels = ENTITY_LABEL_NAMES
        self.per_word_count, self.all_words_count, self.all_sentences = word_counter('recipes.txt')
        self.recipe_model = word2vec.Word2Vec.load(
            'save/recipe.model'
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_label = self.entity_labels[idx]

        positive_rules_ebds = list()
        positive_labels = list()

        positive_file = open(self.rules_dir + positive_label + '_test.txt', mode='r')

        for positive_line in positive_file:
            positve_num, positive_rule = positive_line.strip('\n').split('\t\t')
            positive_rule_ebd = sentence2vec_cat(self.recipe_model, positive_rule)
            positive_rules_ebds.append(torch.tensor(positive_rule_ebd))
            positive_labels.append(idx)

        positive_file.close()
        positive_rules_ebds = torch.tensor(positive_rules_ebds)
        positive_labels = torch.tensor(positive_labels)
        return positive_rules_ebds, positive_labels

    @staticmethod
    def collate_fn(data):
        print(data)
        return data


class TAPIterator(object):
    def __init__(self, train_tap_dataloader):
        self.iterator = self.one_shot_iterator(train_tap_dataloader)

    def __next__(self):
        data = next(self.iterator)

        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
