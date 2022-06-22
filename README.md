Use Anaconda  to Create environment 
---
1. Conda load environment.yml file 
```
conda env create --file d4.yml -n env_name
```

-----


Run the commands on anaconda :
--



1. Install Pytorch and Torchvision with cudatoolkit.

**Run** - Linux

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

For other versions: https://pytorch.org/

2. Install open-cv , gcc & g++ ≥ 5.4 ,ninja

3. Install detectron-2

**Run**
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

4. Clone the detectron-2 repository.

**Run**
```
git clone https://github.com/facebookresearch/detectron2.git
```

5. Keep the detectron-2 repo in your current working folder.


6. Run this script 

**Run**
```
python3 detect.py --input_path /media/sabrina/Work/detec_test/input --extend_path /media/sabrina/Work/detec_test/output --input_format jpg --output_format jpg
```
>>pwd of your Input folder and paste the path in *--input_path*

>>pwd of your Output folder and paste the path in *--extend_path*




---

This script will :

Detect the object using Detectron-2 from input image and add padding when required ; to  output all images with aspect ratio 0.75.

