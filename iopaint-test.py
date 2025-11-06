import PIL
import time
from pathlib import Path

# Copyright [2024] [Sanster]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Import the function you want to use
from iopaint.processing import single_inpaint

# --- Your Configuration ---
# BASE_DIR = "APPR3/bottom"
BASE_DIR = "TESTDATA"

IMAGE_NAME = "pano8192x4096_bottom.png"
MASK_NAME = "mask8192x4096_bottom_blur3.png"


# --- Construct Paths using pathlib.Path ---
# batch_inpaint expects Path objects
base_path = Path(BASE_DIR)
input_path = base_path / IMAGE_NAME
mask_path = base_path / MASK_NAME
output_path = base_path / "out" / "out2.png"

start_time = time.time()

result = single_inpaint(
    input_path,
    mask_path,
)
end_time = time.time()
total_seconds = int(end_time - start_time)

# Save the result
image = PIL.Image.fromarray(result)
output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
image.save(output_path)


print(f"\n--- Function finished in {total_seconds} seconds ---")

