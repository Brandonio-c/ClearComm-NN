import os
import argparse
import json
import time
import soundfile as sf
import torch
from asteroid.models import ConvTasNet
from asteroid.utils import tensors_to_device

scheduler = None

model_path_48kHz = os.path.join("/vulcanscratch", "azheng15", "spring2024", "CMSC720", "asteroid", "egs", "2023_dns_challenge", "ConvTasNet", "exp", "best", "best_model.pth")
model = ConvTasNet.from_pretrained(model_path_48kHz)

file_48kHz = os.path.join("/vulcanscratch", "azheng15", "spring2024", "CMSC720", "DNS-Challenge", "test_001", "Original_audio.mp3")
output_48kHz = os.path.join("/vulcanscratch", "azheng15", "test_data", "speed_testing", "model_48kHz.mp3")

signal, sample_rate = sf.read(file_48kHz)

# signal = tensors_to_device([signal], device=model_device)
signal = torch.Tensor(signal[:,0][:48000])

start = time.time()
result = model(signal)
end = time.time()

print(result[0].detach().numpy().shape)

sf.write(output_48kHz, result[0].detach().numpy(), 48000)

print("Total time needed 48k for 1 second: ", end - start)

#%%
audio_file = os.path.join("/vulcanscratch", "azheng15", "spring2024", "CMSC720", "DNS-Challenge", "test_001", "Original_audio.mp3")

model_path_8kHz = os.path.join("/vulcanscratch", "azheng15", "spring2024", "CMSC720", "asteroid", "egs", "librimix", "ConvTasNet", "exp", "train_convtasnet_LibriMix_ConvTasNet_enh_single",  "best_model.pth")
model = ConvTasNet.from_pretrained(model_path_8kHz)

file_48kHz = os.path.join("/vulcanscratch", "azheng15", "spring2024", "CMSC720", "DNS-Challenge", "test_001", "Original_audio.mp3")
output_8kHz = os.path.join("/vulcanscratch", "azheng15", "test_data", "speed_testing", "model_8kHz.mp3")

signal, sample_rate = sf.read(file_48kHz)

# signal = tensors_to_device([signal], device=model_device)
signal = torch.Tensor(signal[:,0][:48000])

start = time.time()
result = model(signal)
end = time.time()

print(result[0].detach().numpy().shape)

sf.write(output_8kHz, result[0].detach().numpy() , 48000)

print("Total time needed 8k for 1 second: ", end - start)

#%%

audio_file = os.path.join("/vulcanscratch", "azheng15", "spring2024", "CMSC720", "DNS-Challenge", "test_001", "Original_audio.mp3")

model_path_16kHz = os.path.join("/vulcanscratch", "azheng15", "test_data", "pytorch_model.bin")   
model = ConvTasNet.from_pretrained(model_path_16kHz)

file_48kHz = os.path.join("/vulcanscratch", "azheng15", "spring2024", "CMSC720", "DNS-Challenge", "test_001", "Original_audio.mp3")
output_16kHz = os.path.join("/vulcanscratch", "azheng15", "test_data", "speed_testing", "model_16kHz.mp3")

signal, sample_rate = sf.read(file_48kHz)

# signal = tensors_to_device([signal], device=model_device)
signal = torch.Tensor(signal[:,0][:48000])

start = time.time()
result = model(signal)
end = time.time()

print(result[0].detach().numpy().shape)

sf.write(output_16kHz, result[0].detach().numpy() , 48000)

print("Total time needed 16k for 1 second: ", end - start)



    

