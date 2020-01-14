#!/usr/bin/python
# encoding: utf-8

import torch

class Alphabet:
	def __init__(self):
		common = '''0123456789!?"'#$%&@()[]{}<>+-*/\\.,:;^~=|_'''
		english = 'abcdefghijklmnopqrstuvwxyz'
		russian = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

		self.chars = common + english + russian

		self.dict = {}
		for i, char in enumerate(self.chars):
			self.dict[char] = i

		ENGLISH = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
		RUSSIAN = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'

		for i in range(len(ENGLISH)):
			self.dict[ENGLISH[i]] = self.dict[english[i]]
		for i in range(len(RUSSIAN)):
			self.dict[RUSSIAN[i]] = self.dict[russian[i]]

		self.dict['«'] = self.dict['"']
		self.dict['»'] = self.dict['"']

	def encode(self, words):
		label_size = torch.LongTensor([len(word) for word in words])
		label = torch.zeros([len(words), max(label_size)], dtype=torch.long)
		for i, word in enumerate(words):
			label[i, :len(word)] = torch.LongTensor([self.dict[char] + 1 for char in word])
		return label, label_size

	def decode(self, pred, pred_size):
		words = []
		for encoded_word, encoded_word_length in zip(pred, pred_size):
			word = ''
			for i in range(encoded_word_length):
				if encoded_word[i] != 0 and (not (i > 0 and encoded_word[i - 1] == encoded_word[i])):
					word += self.chars[encoded_word[i] - 1]
			words.append(word)
		return words

alphabet = Alphabet()
