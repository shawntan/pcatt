# aoatt

Install dependencies for C++ code
```
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/96aa5993-5b22-4a9b-91ab-da679f422594/intel-oneapi-base-toolkit-2025.0.0.885_offline.sh
sudo sh ./intel-oneapi-base-toolkit-2025.0.0.885_offline.sh -a --cli
cd <install_dir>
```
Initialize environment variables
```
cd <install_dir>
. ./oneapi/tbb/latest/env/vars.sh
```

```
c++ -O3 -Wall -shared -std=c++20 -ltbb -fPIC $(python3 -m pybind11 --includes) greedy_builder.cpp -o greedy_builder$(python3-config --extension-suffix)
```