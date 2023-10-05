import os
import yaml
import torchaudio
import matplotlib.patches as mpatches

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def plot(x, y):
    x = TSNE(n_components=3, learning_rate='auto', perplexity=30).fit_transform(x)
    colors = ['red', 'pink', 'green', 'orange', 'blue']
    emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
    handles = []
    for idx, color in enumerate(colors):
        handles.append(mpatches.Patch(color=color, label=emotions[idx]))

    y_colors = [colors[label] for label in y]
    fig = plt.figure(figsize=(20, 14))
    ax = plt.axes(projection='3d')
    ax.scatter3D(x[:,0], x[:,1], x[:,2], color=y_colors, s=300)
    plt.legend(handles=handles)
    plt.show()


def load_filepaths(filename, split=','):
    with open(filename, encoding='utf-8') as f:
        next(f)     # Remove header line
        metadata = {
            'audio': [],
            'emotion': [],
        }
        for line in f:
            path, emo = line.strip().split(split)
            metadata['audio'].append(path)
            metadata['emotion'].append(emo)

    return metadata


def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load(stream, Loader=yaml.FullLoader)
    return docs


def load_metadata(data_path, speaker_id):
    emo_ch_en = {
        '生气': 'Angry',
        '快乐': 'Happy',
        '中立': 'Neutral',
        '伤心': 'Sad',
        '惊喜': 'Surprise'
    }
    speaker_id = f'{speaker_id:04d}'
    speaker_path = os.path.join(data_path, speaker_id)
    metadata_file = os.path.join(speaker_path, speaker_id +'.txt')
    metadata = {
        'audio': [],
        'emotion': []
    }
    with open(metadata_file, 'r', encoding='utf8') as f:
        for line in f:
            record = line.strip().split('\t')
            if record[2] in emo_ch_en:
                record[2] = emo_ch_en[record[2]]
            metadata['audio'].append(f'{speaker_path}/{record[2]}/{record[0]}.wav')
            metadata['emotion'].append(record[2])
    return metadata


class Dotdict(dict):
    
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class Hparam(Dotdict):

    def __init__(self, file='config.yaml'):
        super(Dotdict, self).__init__()
        hp_dict = load_hparam(file)
        hp_dotdict = Dotdict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)
            
    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__