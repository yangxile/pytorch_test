import torch
from torchvision import models

torch.manual_seed(0)
sample_input = [torch.rand(1, 3, 224, 224)]
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.to(memory_format=torch.channels_last) # Added by Arm
model = model.eval()

import numpy as np

expected_output = [-0.98073,  0.01846778, -0.39239907,  0.89609253, -0.30209816]
model_output = model(*sample_input)[:,:5].detach().numpy()

output_no_loss = np.allclose(model_output, expected_output, atol=1e-5)
print('Output features the same?', output_no_loss)

import time

with torch.no_grad():
    # warmup runs
    for i in range(10):
        model(*sample_input)

    # run 100 times
    times = []
    for i in range(1000):
      start_time = time.time()

      model(*sample_input)

      end_time = time.time()
      measured_time = end_time - start_time
      times.append(measured_time)

import statistics
avg_time = sum(times) / len(times)
stddev_time = statistics.stdev(times)

results = {
  "output_no_loss": output_no_loss,
  "avg_time": avg_time,
  "stddev_time": stddev_time
}

print(results)