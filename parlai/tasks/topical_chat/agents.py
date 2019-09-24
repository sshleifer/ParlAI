
from parlai.core.teachers import ParlAIDialogTeacher
import copy



from parlai.core.agents import create_task_agent_from_taskname
from parlai.core.teachers import FixedDialogTeacher
from .build import build

import json
import os
import random


TOKEN_NOCHOSEN = 'no_passages_used'
TOKEN_KNOWLEDGE = '__knowledge__'
DATAPATH = 'alexa-prize-topical-chat-dataset/conversations/train.json'

def _first_val(dictionary):
    vals = list(dictionary.values())
    if len(vals) > 0:
        return vals[0]
    return ''


def _first_key(dictionary):
    keys = list(dictionary.keys())
    if len(keys) > 0:
        return keys[0]
    return ''


def _get_chosen_title_and_sent(wizard_entry, k_dict):
    """
    Return a nicely extracted title and chosen sentence.
    :return: pair (title, sentence)
    """
    title_dict = wizard_entry.get('checked_passage', 'none')
    sentence_dict = wizard_entry.get('checked_sentence', {})
    title = None
    sentence = None
    if sentence_dict == {}:
        title = sentence = TOKEN_NOCHOSEN
    else:
        sentence = _first_val(sentence_dict)
        if sentence == TOKEN_NOCHOSEN:
            title = TOKEN_NOCHOSEN
        else:
            title = ''
            # cand_title1 is the title from the `checked_passage`
            cand_title1 = _first_val(title_dict) if title_dict else ''
            # cand_title2 is the extracted title of the passage from the
            #   sentence dict, which is e.g. `self_Vermont_Syrup_0`
            cand_title2 = ' '.join(_first_key(sentence_dict).split('_')[1:-1])
            if (
                cand_title1
                and cand_title1 in k_dict
                and sentence in k_dict[cand_title1]
            ):
                title = cand_title1
            elif cand_title2 in k_dict and sentence in k_dict[cand_title2]:
                title = cand_title2
            else:  # neither candidate title is the right one
                for t, passage in k_dict.items():
                    if sentence in passage:
                        title = t
                        break

    return title, sentence


def _path(opt, split='random_split'):
    #assert os.path.exists(DATAPATH), DATAPATH
    build(opt)
    path = os.path.join(opt['datapath'], 'topical_chat', 'train.json')
    return path

    build(opt)
    dp = os.path.join(opt['datapath'], 'wizard_of_wikipedia')
    dt = opt.get('datatype', 'train').split(':')[0]
    if dt == 'train':
        df = 'train.json'
    else:
        df = '{}_{}.json'.format(dt, split)
    return os.path.join(dp, df)


class DefaultTeacher(FixedDialogTeacher):
    """The default teacher; essentially reads the json file and outputs the
       raw data.

       Actions have the following form:
       {
           'wizard_eval': <evaluation of wizard>,
           'chosen_topic': <chosen_topic>,
           'chosen_topic_passage': <chosen topic passage>,
           'mtdo': <whether the conversation had sufficient overlap>,
           'text': <text>
           'retrieved_topics': <topics retrieved for text>
           'full_retrieved_passages': <full retrieved passages>
           'retrieved_passages': <passages shown to turker>
           'checked_sentence': <checked sentence if wizard, else None>
           'checked_passage': <checked_passage if wizard, else None>
       }

       The 'passages' are lists of 1 entry dicts, mapping a topic to the sentences

       Specify the valid/test split after the last colon in the task, e.g.
       wizard_of_wikipedia:<teacher>:random_split
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        task = opt.get('task', 'wizard_of_wikipedia:WizardOfWikipedia:random_split')
        split = task.split(':')
        split = split[2] if len(split) == 3 else 'random_split'
        opt['task'] = 'topical_chat'
        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self.data_path = _path(opt, split=split)
            self._setup_data()
        self.num_exs = sum(len(v['content']) for k,v in self.data.items())
        self.reset()

    def _setup_data(self):
        print('loading: ' + self.data_path)
        with open(self.data_path) as f:
            self.data = json.load(f)
        self.episode_id_map = dict(enumerate(self.data.keys()))

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return self.num_exs

    def get(self, episode_idx, entry_idx=0):
        d = self.data[self.episode_id_map[episode_idx]]
        dialog_entry = d['content'][entry_idx]
        episode_done = entry_idx == len(d['content']) - 1
        action = {
            #'wizard_eval': d['wizard_eval'],
            #'chosen_topic': d['chosen_topic'],
            #'chosen_topic_passage': d['chosen_topic_passage'],
            # 'id'
            'text': dialog_entry['message'],
            'agent': dialog_entry['agent'],
            'knowledge_source': dialog_entry['knowledge_source'],
            'turn_rating': dialog_entry['turn_rating'],
            'agent': dialog_entry['agent'],
            'episode_done': episode_done,
        }

        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared
