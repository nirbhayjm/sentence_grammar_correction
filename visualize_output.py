import numpy as np
import shelve
# from color import color

def savePredictions(X,y_true_list,y_pred_list,lengths):
	tokDict = shelve.open("map_w2v/uid_dict")

	outputFile = open("savedPredictions.txt","w")

	for i in range(len(lengths)):
		length = lengths[i]
		text_indices = X[i,:length]
		y_t = y_true_list[i,:length]
		y_p = y_pred_list[i,:length]

		# print "text_indices:",text_indices
		
		tokens = map(lambda x: tokDict[str(x)],text_indices)
		# print "Tokens:"," ".join(tokens),"\n\n"

		# toks_true = [ x[0]+"/"+str(x[1]) for x in zip(tokens,y_t) ]
		# print "Tokens_true:"," ".join(toks_true),"\n\n"
		# toks_pred = [ x[0]+"/"+str(x[1]) for x in zip(tokens,y_p) ]

		toks_true = []
		toks_pred = []
		for j,tok in enumerate(tokens):
			if y_t[j] == 2:
				true_tok = "[" + tok + "]"
			else:
				true_tok = tok
			toks_true.append(true_tok)

			if y_p[j] == 2:
				pred_tok = "{" + tok + "}"
			else:
				pred_tok = tok
			toks_pred.append(pred_tok)

		outputFile.write("True:"+" ".join(toks_true)+"\n\n")
		outputFile.write("Pred:"+" ".join(toks_pred)+"\n\n")
		outputFile.write("---\n")

	outputFile.close()
	return
