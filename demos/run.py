import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt

import SpeechRecognizer
import sr_data_utils
import sr_model_utils

libri_path = './data/LibriSpeech/dev-clean'

# the function gets the path to the directory where the files are stored.
# it iterates the dir and subdirs and then processes the audios
# using the 'audioToInputVector' function
# and loads the corresponding text to each audio file.
txts, audios, audio_paths = sr_data_utils.load_data(libri_path, how_many=0)

### Save to & load from .pkl

# # ## to avoid having to process the texts and audios each time we can save them for later use.
sr_data_utils.write_pkl('./pickles/dev_txts.pkl', txts)
sr_data_utils.write_pkl('./pickles/dev_audio_paths.pkl', audio_paths)
sr_data_utils.save_as_pickled_object(audios, './pickles/dev_audios.pkl')

# and load them.
txts = sr_data_utils.load_pkl('./pickles/dev_txts.pkl')
audio_paths = sr_data_utils.load_pkl('./pickles/dev_audio_paths.pkl')
audios = sr_data_utils.load_pkl('./pickles/dev_audios.pkl')

### Explore
print(len(txts), len(audios))

print(txts[0], audios[0])

print([a.shape for a in audios[:10]])

# length in characters.
print([len(t) for t in txts[:10]])


def play_audio(path):
    """Plays the audio of the give path
       within Jupyter Notebook
    """
    samples, sample_rate = librosa.load(path, mono=True, sr=None)
    return ipd.Audio(samples, rate=sample_rate)


play_audio(audio_paths[1])

print(txts[1])

sr_data_utils.plot_wave(audio_paths[1])

sr_data_utils.plot_melspectogram(audio_paths[0])

### Process texts

# the 'process_txts' function calls 'split_txts', 'create_lookup_dicts' and 'convert_txt_to_inds' internally.
specials = ['<EOS>', '<SOS>', '<PAD>']
txts_splitted, unique_chars, char2ind, ind2char, txts_converted = sr_data_utils.process_txts(txts, specials)

print(len(char2ind), len(ind2char))

# before conversion
print('First txt:', txts[0])

# splitted
print('First txt splitted:', txts_splitted[0])

# after conversion
print('First txt converted:', txts_converted[0])

# converted back.
# seems to have worked well.
# Note: <SOS> and <EOS> token were added at start and end of txt.
print('First txt converted back:', sr_data_utils.convert_inds_to_txt(txts_converted[0], ind2char))

# write lookup dicts to .pkl for later use.
# sr_data_utils.write_pkl('./pickles/sr_word2ind.pkl', word2ind)
# sr_data_utils.write_pkl('./pickles/sr_ind2word.pkl', ind2word)

### Sort

# The audios in the dataset differ massively in length. In order to simplify the training process for the model and due to, again, limited resources we will sort them by length and use rather short ones.
# Furthermore the model seems to train better, when first feeded rather short examples.


for t, a in zip(txts[:30], audios[:30]):
    print(len(t), a.shape[0])

# Sort texts by text length or audio length from shortest to longest.
# To keep everything in order we also sort the rest of the data.
txts, audios, audio_paths, txts_splitted, txts_converted = sr_data_utils.sort_by_length(audios,
                                                                                        txts,
                                                                                        audio_paths,
                                                                                        txts_splitted,
                                                                                        txts_converted,
                                                                                        by_text_length=False)

for t, a in zip(txts[:30], audios[:30]):
    print(len(t), a.shape[0])

# Quicklook at length distributions.
# obviously they correlate well with each other.
plt.figure(figsize=(15, 10))
plt.subplot(221)
plt.hist([a.shape[0] for a in audios], bins=100)
plt.title('Distribution of audio lenghts')

plt.subplot(222)
plt.hist([len(t) for t in txts_converted], bins=100)
plt.title('Distribution of text lengths')

## Model

# Now we can start training our the model.

### Train

# Create SpeechRecognizer, build the graph and train the network.


# nice extension ==> avoids having to restart kernel everytime changes are made.

sr_model_utils.reset_graph()
sr = SpeechRecognizer.SpeechRecognizer(char2ind,
                                       ind2char,
                                       './models/models_1000points_0',
                                       num_layers_encoder=2,
                                       num_layers_decoder=2,
                                       rnn_size_encoder=450,
                                       rnn_size_decoder=450,
                                       embedding_dim=10,
                                       batch_size=40,
                                       epochs=500,
                                       use_cyclic_lr=True,
                                       learning_rate=0.00001,
                                       max_lr=0.00003,
                                       learning_rate_decay_steps=700)

sr.build_graph()
sr.train(audios[:1000], txts_converted[:1000], restore_path='./models/models_1000points_0')

# we are terribly overfitting here. therefore it won't generalize well.
# Note: hidden training process. Loss decreased to about 0.4

### Test

sr_model_utils.reset_graph()
sr = SpeechRecognizer.SpeechRecognizer(char2ind,
                                       ind2char,
                                       './models/models_1000points_0',
                                       num_layers_encoder=2,
                                       num_layers_decoder=2,
                                       rnn_size_encoder=450,
                                       rnn_size_decoder=450,
                                       mode='INFER',
                                       embedding_dim=10,
                                       batch_size=1,
                                       beam_width=5)

sr.build_graph()
preds = sr.infer(audios[0:500:20], './models/models_1000points_0')

# preds2txt converts the predictions to text and removes <EOS> and <SOS> tags.
preds_converted = sr_data_utils.preds2txt(preds, ind2char, beam=True)

# prints the created texts side by side with the actual texts and
# prints out an accuracy score of how good the prediction was.
# if the created text is shorter than the actual one we
# penalize by subtracting 0.3 (pretty hard!)
# the accuracy clearly suffers, as the sequences get longer.
sr_data_utils.print_samples(preds_converted, txts_splitted[0:500:20])

## Conclusion

# Sequence to sequence models are really powerful and perform well on speech. However, here I could again only train our model on a really limited amount of data. It would be really interesting to scale this up and see how good it can get. Furthermore I will add a language model to support this model.
#
# To sum up, speech recognition is already an important part and is going to play an even more crucial role in human-machine interaction in the near future. Therefore I am really curious to explore that topic more in depth.
# (with hopefully more computing power)
