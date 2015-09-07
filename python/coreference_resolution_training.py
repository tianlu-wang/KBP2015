from coref_feature_extraction import *

import xml.etree.cElementTree as ET
import os
import cPickle
from random import shuffle
import math

import nltk

ie_folder="data/ie_data/ace2005"

#Main for ACE
event_mention_list=list()
for one_xml_file in os.listdir(ie_folder):
    if ".xml" not in one_xml_file:
        continue
    root=ET.parse(os.path.join(ie_folder,one_xml_file)).getroot()
    for one_event in root[0].findall("event"):
        for one_event_mention in one_event.findall("event_mention"):
            event_mention_list.append((one_event,one_event_mention))

print len(event_mention_list)

training_list=list()
training_coref=list()
training_noncoref=list()

no_flag=0
for i in range(0,len(event_mention_list)):
    for j in range(i+1,len(event_mention_list)):
        event_1_type=".".join([event_mention_list[i][0].attrib['TYPE'],event_mention_list[i][0].attrib['SUBTYPE']])
        event_2_type=".".join([event_mention_list[j][0].attrib['TYPE'],event_mention_list[j][0].attrib['SUBTYPE']])
        if event_1_type != event_2_type:
            continue
        event_mention_1=event_mention_list[i][1]
        event_mention_2=event_mention_list[j][1]
        event_mention_1_id=event_mention_1.attrib['ID']
        event_mention_2_id=event_mention_2.attrib['ID']
        file_1=event_mention_1_id.split('-')[0]
        file_2=event_mention_2_id.split('-')[0]
        if file_1!=file_2:
            continue
        event_1=event_mention_list[i][0]
        event_1_id=event_1.attrib['ID']
        event_2=event_mention_list[j][0]
        event_2_id=event_2.attrib['ID']
        try:
            temp_dict=dict()
            temp_dict["type_subtype"]=extract_type_subtype(event_1)
            temp_dict["trigger_pair"]=extract_trigger_pair(event_mention_1, event_mention_2)
            temp_dict["pos_pair"]=extract_pos_pair(event_mention_1, event_mention_2)
            temp_dict["nominal"]=extract_nominal(event_mention_1)
            temp_dict["nom_number"]=extract_nom_number(event_mention_1)
            temp_dict["pronominal"]=extract_pronominal(event_mention_1)
            temp_dict["exact_match"]=extract_exact_match(event_mention_1,event_mention_2)
            temp_dict["stem_match"]=extract_stem_match(event_mention_1,event_mention_2)
            temp_dict["trigger_sim"]=extract_trigger_sim(event_mention_1,event_mention_2)*5
            temp_dict["entity_exact_match"]=extract_entity_exact_match(event_mention_1,event_mention_2,ie_folder)
            temp_dict["mod"]=extract_mod(event_1)
            temp_dict["pol"]=extract_pol(event_1)
            temp_dict["gen"]=extract_gen(event_1)
            temp_dict["ten"]=extract_ten(event_1)
            temp_dict["mod_conflict"]=extract_mod_conflict(event_1, event_2)
            temp_dict["pol_conflict"]=extract_pol_conflict(event_1, event_2)
            temp_dict["gen_conflict"]=extract_gen_conflict(event_1, event_2)
            temp_dict["ten_conflict"]=extract_ten_conflict(event_1, event_2)
            print no_flag
            if event_1_id==event_2_id:
                training_list.append((temp_dict,"coref"))
                training_coref.append((temp_dict,"coref"))
            else:
                training_list.append((temp_dict,"no_coref"))
                training_noncoref.append((temp_dict,"no_coref"))
            no_flag+=1
        except:
            continue

# cPickle.dump(training_list,open("data/coref_training.dat",'w'))
# cPickle.dump(training_coref,open("data/coref.dat",'w'))
# cPickle.dump(training_noncoref,open("data/noncoref.dat",'w'))

# training_list=cPickle.load(open("data/coref_training.dat"))
# training_coref=cPickle.load(open("data/coref.dat"))
# training_noncoref=cPickle.load(open("data/noncoref.dat"))

shuffle(training_noncoref)

group_no=int(math.ceil(float(len(training_noncoref))/len(training_coref)))

classifier_list=list()

for i in range(0, group_no):
    negative_start_index=i*len(training_coref)
    if (i+1)*len(training_coref)>len(training_noncoref):
        negative_end_index=len(training_noncoref)
    else:
        negative_end_index=(i+1)*len(training_coref)
    training_noncoref_sub=training_noncoref[negative_start_index:negative_end_index]
    maxent_classifier=nltk.MaxentClassifier.train(training_coref+training_noncoref_sub)
    classifier_list.append(maxent_classifier)


cPickle.dump(classifier_list,open("data/coref_classifier_model.dat",'w'))