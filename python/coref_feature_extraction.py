__author__ = 'charlesztt'

import xml.etree.cElementTree as ET
import os
import nltk
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

# ie_folder = "data/ie_data/isis_jan_feb"

def return_entity(entity_ref_id, ie_folder):
    tree=ET.parse(os.path.join(ie_folder,return_file_name_with_mention_id(entity_ref_id)+'.apf.xml'))
    root=tree.getroot()
    for entity in root[0].findall('entity'):
        if entity.attrib['ID'] == '-'.join(entity_ref_id.split('-')[0:2]):
            return entity
    for value in root[0].findall('value'):
        if value.attrib['ID'] == '-'.join(entity_ref_id.split('-')[0:2]):
            return value
    for timex2 in root[0].findall('timex2'):
        if timex2.attrib['ID'] == '-'.join(entity_ref_id.split('-')[0:2]):
            return timex2

def share_token(string1,string2):
    string1=string1.replace("\n"," ")
    string2=string2.replace("\n"," ")
    string1_list=string1.split()
    string2_list=string2.split()
    f=open('./data/stopword.txt')
    for line in f:
        line=line.replace("\n","")
        stopword_list=line.split(',')
    f.close()
    for one_token in string1_list:
        if one_token in stopword_list:
            continue
        if one_token in string2_list:
            return True
    for one_token in string2_list:
        if one_token in stopword_list:
            continue
        if one_token in string1_list:
            return True
    return False

def share_token_in_entity(entity1,entity2):
    f=open('./data/stopword.txt')
    for line in f:
        line=line.replace("\n","")
        stopword_list=line.split(',')
    f.close()
    entity_mention_list_1=list()
    for one_entity in entity1.findall(entity1.tag+'_mention'):
        entity_mention_list_1.append(one_entity[0][0].text.replace("\n"," "))
    entity_mention_list_2=list()
    for one_entity in entity2.findall(entity1.tag+'_mention'):
        entity_mention_list_2.append(one_entity[0][0].text.replace("\n"," "))
    for one_entity_metion in entity_mention_list_1:
        one_entity_metion_list=one_entity_metion.split()
        for another_entity_mention in entity_mention_list_2:
            another_entity_mention_list=another_entity_mention.split()
            for one_token in one_entity_metion_list:
                if one_token in stopword_list:
                    continue
                if one_token in another_entity_mention_list:
                    return True
    return False

def return_file_name_with_object_id(id):
    return "-".join(id.split("-")[:-1])

def return_file_name_with_mention_id(id):
    return "-".join(id.split("-")[:-2])

#features
def extract_type_subtype(one_event):
    return ".".join([one_event.attrib['TYPE'],one_event.attrib['SUBTYPE']])

def extract_trigger_pair(event_mention_1, event_mention_2):
    trigger1=""
    trigger2=""
    for one_anchor in event_mention_1.findall("anchor"):
        trigger1=one_anchor[0].text
    for one_anchor in event_mention_2.findall("anchor"):
        trigger2=one_anchor[0].text
    return (trigger1, trigger2)

def extract_pos_pair(event_mention_1, event_mention_2):
    trigger1=""
    extent1=""
    trigger2=""
    extent2=""
    for one_anchor in event_mention_1.findall("anchor"):
        trigger1=one_anchor[0].text
    for one_anchor in event_mention_2.findall("anchor"):
        trigger2=one_anchor[0].text
    for one_extent in event_mention_1.findall("extent"):
        extent1=one_extent[0].text
    for one_extent in event_mention_2.findall("extent"):
        extent2=one_extent[0].text
    text1 = nltk.word_tokenize(extent1)
    dict1 = nltk.pos_tag(text1)
    for one_pair in dict1:
        if one_pair[0] in trigger1 or trigger1 in one_pair[0]:
            pos1=one_pair[1]
            break
    text2 = nltk.word_tokenize(extent2)
    dict2 = nltk.pos_tag(text2)
    for one_pair in dict2:
        if one_pair[0] in trigger2 or trigger2 in one_pair[0]:
            pos2=one_pair[1]
            break
    return (pos1, pos2)

def extract_nominal(one_event_mention):
    trigger=""
    extent=""
    for one_anchor in one_event_mention.findall("anchor"):
        trigger = one_anchor[0].text
    for one_extent in one_event_mention.findall("extent"):
        extent = one_extent[0].text
    text = nltk.word_tokenize(extent)
    dict = nltk.pos_tag(text)
    for one_pair in dict:
        if one_pair[0] in trigger or trigger in one_pair[0]:
            pos=one_pair[1]
            break
    return "NN" in pos

def extract_nom_number(one_event_mention):
    trigger=""
    extent=""
    for one_anchor in one_event_mention.findall("anchor"):
        trigger = one_anchor[0].text
    for one_extent in one_event_mention.findall("extent"):
        extent = one_extent[0].text
    text = nltk.word_tokenize(extent)
    dict = nltk.pos_tag(text)
    for one_pair in dict:
        if one_pair[0] in trigger or trigger in one_pair[0]:
            pos=one_pair[1]
            break
    if "NN" in pos:
        if pos=="NNS":
            return "plural"
        else:
            return "singular"
    else:
        return None

def extract_pronominal(one_event_mention):
    trigger=""
    extent=""
    for one_anchor in one_event_mention.findall("anchor"):
        trigger = one_anchor[0].text
    for one_extent in one_event_mention.findall("extent"):
        extent = one_extent[0].text
    text = nltk.word_tokenize(extent)
    dict = nltk.pos_tag(text)
    for one_pair in dict:
        if one_pair[0] in trigger or trigger in one_pair[0]:
            pos=one_pair[1]
            break
    return pos == "PRP"

def extract_exact_match(event_mention_1, event_mention_2):
    trigger1=""
    trigger2=""
    for one_anchor in event_mention_1.findall("anchor"):
        trigger1=one_anchor[0].text
    for one_anchor in event_mention_2.findall("anchor"):
        trigger2=one_anchor[0].text
    return trigger1==trigger2

def extract_stem_match(event_mention_1, event_mention_2):
    trigger1=""
    trigger2=""
    for one_anchor in event_mention_1.findall("anchor"):
        trigger1=one_anchor[0].text
    for one_anchor in event_mention_2.findall("anchor"):
        trigger2=one_anchor[0].text
    stemmer = PorterStemmer()
    return stemmer.stem(trigger1.split(" ")[0]) == stemmer.stem(trigger2.split(" ")[0])

def extract_trigger_sim(event_mention_1, event_mention_2):
    trigger1=""
    trigger2=""
    for one_anchor in event_mention_1.findall("anchor"):
        trigger1=one_anchor[0].text
    for one_anchor in event_mention_2.findall("anchor"):
        trigger2=one_anchor[0].text
    wnl = WordNetLemmatizer()

    try:
        point1=wn.synsets(wnl.lemmatize(trigger1))[0]
        point2=wn.synsets(wnl.lemmatize(trigger2))[0]
        if point1.path_similarity(point2) is None:
            return 0.0
        else:
            return point1.path_similarity(point2)
    except:
        return 0.0


def extract_entity_exact_match(event_mention_1, event_mention_2, ie_folder):
    event_mention_argument_list_1=list()
    event_mention_argument_list_2=list()
    for one_event_mention_argument in event_mention_1.findall('event_mention_argument'):
        event_mention_argument_list_1.append(one_event_mention_argument)
    for one_event_mention_argument in event_mention_2.findall('event_mention_argument'):
        event_mention_argument_list_2.append(one_event_mention_argument)
    if len(event_mention_argument_list_2)+len(event_mention_argument_list_1)==0:
            return False
    no_flag=0
    for event_mention_argument_a in event_mention_argument_list_1:
        for event_mention_argument_b in event_mention_argument_list_2:
            if event_mention_argument_a.attrib['ROLE']==event_mention_argument_b.attrib['ROLE']:
                no_flag+=1
    if no_flag==0:
        return False
    for event_mention_argument_a in event_mention_argument_list_1:
        for event_mention_argument_b in event_mention_argument_list_2:
            if event_mention_argument_a.attrib['ROLE']==event_mention_argument_b.attrib['ROLE']:
                entity_mention_1=return_entity(event_mention_argument_a.attrib['REFID'],ie_folder)
                entity_mention_2=return_entity(event_mention_argument_b.attrib['REFID'],ie_folder)
                if entity_mention_1 == None or entity_mention_2 == None:
                    return False
                    continue
                if share_token_in_entity(entity_mention_1,entity_mention_2)==False:
                    return False
    return True


def extract_mod(one_event):
    return one_event.attrib['MODALITY']

def extract_pol(one_event):
    return one_event.attrib['POLARITY']

def extract_gen(one_event):
    return one_event.attrib['GENERICITY']

def extract_ten(one_event):
    return one_event.attrib['TENSE']

def extract_mod_conflict(event_1,event_2):
    if extract_mod(event_1) == extract_mod(event_2):
        return False
    else:
        return True

def extract_pol_conflict(event_1,event_2):
    if extract_pol(event_1) == extract_pol(event_2):
        return False
    else:
        return True

def extract_gen_conflict(event_1,event_2):
    if extract_gen(event_1) == extract_gen(event_2):
        return False
    else:
        return True

def extract_ten_conflict(event_1,event_2):
    if extract_ten(event_1) == extract_ten(event_2):
        return False
    else:
        return True