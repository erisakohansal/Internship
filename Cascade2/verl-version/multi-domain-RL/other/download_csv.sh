#!/bin/bash

NEMO_GYM_COMMIT=5675816c20454ea9449eb47ea3935605b4d7461e

curl -L "https://github.com/NVIDIA-NeMo/Gym/archive/${NEMO_GYM_COMMIT}.tar.gz" \
  | tar -xz --strip-components=3 \
    "Gym-${NEMO_GYM_COMMIT}/resources_servers/workplace_assistant/csv_data"