__author__ = 'charlesztt'

import cPickle
import datetime

from coref_feature_extraction import *

def extract_date(date_8_digit):
    year=int(date_8_digit[0:4])
    month=int(date_8_digit[4:6])
    day=int(date_8_digit[6:8])
    return datetime.date(year, month, day).timetuple()

######Change: change the input ie folder of your testing results
ie_folder="./data/ie_data/isis"

#edl_expansion_coreference_groundtruth_list=list()
event_mention_list=list()

for one_file in os.listdir(ie_folder):#os.listdir('../../data/annotation_data'):
    if '.xml' not in one_file:
        continue
    tree=ET.parse(os.path.join(ie_folder,one_file))
    root = tree.getroot()
    for one_event in root[0].findall('event'):
        for one_event_mention in one_event.findall('event_mention'):
            event_mention_list.append((one_event,one_event_mention))

testing_list=list()

no_flag=0

######Change: change the event id record file
f=open("data/isis_event_id.txt","w")
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
        event_mention_1_date=event_mention_1.attrib["ID"].split("_")[-3]
        event_mention_2_date=event_mention_2.attrib["ID"].split("_")[-3]
        event_mention_1_date_index = extract_date(event_mention_1_date)[7]
        event_mention_2_date_index = extract_date(event_mention_2_date)[7]
        # if abs(event_mention_1_date_index-event_mention_2_date_index)>=7:
        #     continue
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
            no_flag+=1
            print no_flag, event_mention_1_id, event_mention_2_id
            f.write("%s\t%s\n"%(event_mention_1_id, event_mention_2_id))
            testing_list.append(temp_dict)
        except:
            continue
f.close()

######Change: change the dump file of your test features
cPickle.dump(testing_list,open("data/isis.dat",'w'))