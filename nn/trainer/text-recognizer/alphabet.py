#!/usr/bin/python
# encoding: utf-8

import torch

class Alphabet:
	def __init__(self):
		self.char_groups = [
			"0OoОо",
			"1",
			"2",
			"3ЗзЭэ",
			"4",
			"5",
			"6б",
			"7",
			"8",
			"9",
			"!",
			"?",
			"#",
			"$",
			"%",
			"&",
			"@",
			"([{",
			"<",
			")]}",
			">",
			"+",
			"-",
			"*",
			"/",
			"\\",
			".,",
			":;",
			"\"'",
			"^",
			"~",
			"=",
			"|lI",
			"_",
			"AА",
			"aа",
			"BВв",
			"bЬьЪъ",
			"CcСс",
			"D",
			"d",
			"EЕЁ",
			"eеё",
			"F",
			"f",
			"G",
			"g",
			"HНн",
			"h",
			"i",
			"J",
			"j",
			"KКк",
			"k",
			"L",
			"MМм",
			"m",
			"N",
			"n",
			"PpРр",
			"R",
			"r",
			"Q",
			"q",
			"Ss",
			"TТт",
			"t",
			"U",
			"u",
			"Vv",
			"Ww",
			"XxХх",
			"Y",
			"yУу",
			"Zz",
			"Б",
			"Гг",
			"Дд",
			"Жж",
			"ИиЙй",
			"Лл",
			"Пп",
			"Фф",
			"Цц",
			"Чч",
			"ШшЩщ",
			"Ыы",
			"Юю",
			"Яя"
		]

		self.dict = {}
		for code, char_group in enumerate(self.char_groups):
			for char in char_group:
				self.dict[char] = code

	def encode(self, textlines):
		label_size = torch.LongTensor([len(textline) for textline in textlines])
		label = torch.zeros([len(textlines), max(label_size)], dtype=torch.long)
		for i, textline in enumerate(textlines):
			label[i, :len(textline)] = torch.LongTensor([self.dict[char] + 1 for char in textline])
		return label, label_size

alphabet = Alphabet()
