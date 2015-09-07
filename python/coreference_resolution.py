__author__ = 'charlesztt'

import cPickle
import nltk


training_model=cPickle.load(open("./data/coref_classifier_model.dat"))
######Change: the event id record file
f=open("data/isis_event_id.txt")
######Change: the dump file of your test features
testing_set=cPickle.load(open("data/isis.dat"))

######Change: coreference confidenece record file
f_w=open("./data/isis_result.txt","w")
for i in range(0,len(testing_set)):
    one_line=f.readline()
    one_line=one_line.replace("\n","")
    id_pair=one_line.split("\t")
    sum_of_proba=0
    for one_training_model in training_model:
        one_proba=float(one_training_model.prob_classify(testing_set[i]).prob('coref'))
        sum_of_proba+=one_proba
    # print "%s|||%s|||%f\n"%(id_pair[0],id_pair[1],sum_of_proba/len(training_model))
    f_w.write("%s|||%s|||%f\n"%(id_pair[0],id_pair[1],one_proba))

f.close()
f_w.close()