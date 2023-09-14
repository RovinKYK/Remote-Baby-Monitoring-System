import tensorflow as tf
import tflite_model_maker as mm
from tflite_model_maker import audio_classifier
# import tempfile
import os

import numpy as np
import matplotlib.pyplot as plt
from pydub.playback import play
import seaborn as sns

# import itertools
import glob
import random

# from IPython.display import Audio, Image
from scipy.io import wavfile

print(f"TensorFlow Version: {tf.__version__}")
print(f"Model Maker Version: {mm.__version__}")

'''crys_dataset_folder = tf.keras.utils.get_file('crys_dataset.zip',
                                                'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/crys_dataset.zip',
                                                cache_dir='./',
                                                cache_subdir='dataset',
                                                extract=True)'''

data_dir = './dataset'


cry_code_to_name = {
    'bp': 'belly pain',
    'di': 'discomfort',
    'hu': 'hungry',
    'ti': "tired",
    'bu': 'burping'
}

test_files = os.path.join(data_dir, 'test/*/*.wav')


def get_random_audio_file():
    test_list = glob.glob(test_files)
    random_audio_path = random.choice(test_list)
    return random_audio_path

# audio_path = 'a\c\c\e.wav'


def show_cry_data(audio_path):
    sample_rate, audio_data = wavfile.read(audio_path, 'rb')

    cry_code = audio_path.split("\\")[-2]
    print(audio_path)
    print(f'cry code: {cry_code}')
    print(f'cry name: {cry_code_to_name[cry_code]}')

    title = f'{cry_code_to_name.get(cry_code, "Unknown")} ({cry_code})'
    plt.title(title)
    plt.plot(audio_data)
    plt.show()

    '''temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav_file.close()
    temp_wav_filename = temp_wav_file.name
    wavfile.write(temp_wav_filename, sample_rate, audio_data)
    os.system(f'ffplay -nodisp -autoexit "{temp_wav_filename}"')
    os.remove(temp_wav_filename)'''

# show_cry_data(get_random_audio_file())

spec = audio_classifier.YamNetSpec(
    keep_yamnet_and_custom_heads=True,
    frame_step=3 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH,
    frame_length=6 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)

print("Model created")

train_data = audio_classifier.DataLoader.from_folder(
    spec, os.path.join(data_dir, 'train'), cache=True)
train_data, validation_data = train_data.split(0.9)
test_data = audio_classifier.DataLoader.from_folder(
    spec, os.path.join(data_dir, 'test'), cache=True)

print("Dataset Loaded")

batch_size = 10
epochs = 100

model = audio_classifier.create(
    train_data,
    spec,
    validation_data,
    batch_size=batch_size,
    epochs=epochs)

print('Model Trained')


print('Evaluating the model')
model.evaluate(test_data)


def show_confusion_matrix(confusion, test_labels):
    """Compute confusion matrix and normalize."""
    # confusion_normalized = confusion.astype("float") / confusion.sum(axis=1)
    confusion_normalized = confusion.astype("float")
    axis_labels = test_labels
    ax = sns.heatmap(
        confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
        cmap='Blues', annot=True, fmt='.2f', square=True)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


confusion_matrix = model.confusion_matrix(test_data)
show_confusion_matrix(confusion_matrix.numpy(), test_data.index_to_label)
print("Confusion matrix created")

serving_model = model.create_serving_model()

print(f'Model\'s input shape and type: {serving_model.inputs}')
print(f'Model\'s output shape and type: {serving_model.outputs}')

random_audio = get_random_audio_file()
show_cry_data(random_audio)

sample_rate, audio_data = wavfile.read(random_audio, 'rb')

audio_data = np.array(audio_data) / tf.int16.max
input_size = serving_model.input_shape[1]

splitted_audio_data = tf.signal.frame(audio_data, input_size, input_size, pad_end=True, pad_value=0)

print(f'Test audio path: {random_audio}')
print(f'Original size of the audio data: {len(audio_data)}')
print(f'Number of windows for inference: {len(splitted_audio_data)}')

print(random_audio)

results = []
print('Result of the window ith:  your model class -> score,  (spec class -> score)')
for i, data in enumerate(splitted_audio_data):
    yamnet_output, inference = serving_model(data)
    results.append(inference[0].numpy())
    result_index = tf.argmax(inference[0])
    spec_result_index = tf.argmax(yamnet_output[0])
    t = spec._yamnet_labels()[spec_result_index]
    result_str = f'Result of the window {i}: ' \
                 f'\t{test_data.index_to_label[result_index]} -> {inference[0][result_index].numpy():.3f}, ' \
                 f'\t({spec._yamnet_labels()[spec_result_index]} -> {yamnet_output[0][spec_result_index]:.3f})'
    print(result_str)

results_np = np.array(results)
mean_results = results_np.mean(axis=0)
result_index = mean_results.argmax()
print(f'Mean result: {test_data.index_to_label[result_index]} -> {mean_results[result_index]}')

models_path = './Trained_Models'
print(f'Exporing the TFLite model to {models_path}')

model.export(models_path, tflite_filename='my_crys_model.tflite')
model.export(models_path, export_format=[mm.ExportFormat.SAVED_MODEL, mm.ExportFormat.LABEL])
