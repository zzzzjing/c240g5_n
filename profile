#
# CloudLab Profile for c240g5 Node with Tesla P100 GPU
# Ubuntu 22.04 + NVIDIA Driver + CUDA 11.7 + PyTorch
#

import geni.portal as portal
import geni.rspec.pg as pg

# Initialize the Request object
request = portal.context.makeRequestRSpec()

# Request a c240g5 node
node = request.RawPC("node")
node.hardware_type = "c240g5"
node.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU22-64-STD"
node.exclusive = True

# (Optional) Create extra FS space at /mnt
node.addService(pg.Execute(shell="sh", command="sudo /usr/local/etc/emulab/mkextrafs.sh /mnt"))

# 1. System update and Python environment setup
node.addService(pg.Execute(shell="sh", command="sudo apt-get update"))
node.addService(pg.Execute(shell="sh", command="sudo apt-get install -y python3.8 python3.8-venv python3-pip"))

# 2. Install NVIDIA driver
# For Tesla P100 (Pascal), driver 525+ is recommended on Ubuntu 22.04.
# If you want an older driver, you can switch to nvidia-driver-510 or a newer one like 535 if available.
node.addService(pg.Execute(shell="sh", command="sudo apt-get install -y nvidia-driver-525"))

# 3. Install CUDA 11.7
# If you'd like to use another version, e.g. 11.8 or 12.x, adjust accordingly.
node.addService(pg.Execute(shell="sh", command="sudo apt-get install -y cuda-11-7"))
node.addService(pg.Execute(shell="sh", command="sudo ln -s /usr/local/cuda-11.7 /usr/local/cuda || true"))

# 4. Set CUDA environment variables in .bashrc
node.addService(
    pg.Execute(
        shell="sh",
        command=(
            "echo 'export PATH=/usr/local/cuda-11.7/bin:$PATH' >> ~/.bashrc && "
            "echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc"
        )
    )
)

# Force reload of .bashrc changes (for subsequent commands in the same shell)
node.addService(pg.Execute(shell="sh", command="source ~/.bashrc || true"))

# 5. Install PyTorch (with CUDA 11.7) and other Python packages system-wide
node.addService(pg.Execute(
    shell="sh",
    command=(
        "pip3 install --upgrade pip && "
        "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 && "
        "pip3 install numpy scipy pandas matplotlib"
    )
))

# 6. Create a Python virtual environment
node.addService(pg.Execute(shell="sh", command="python3.8 -m venv dl_env"))

# 7. Install deep learning packages in the virtual environment
node.addService(pg.Execute(
    shell="sh",
    command=(
        "source dl_env/bin/activate && "
        "pip install --upgrade pip && "
        "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 && "
        "pip install numpy scipy pandas matplotlib"
    )
))

# 8. Verify CUDA and PyTorch configuration
node.addService(pg.Execute(
    shell="sh",
    command=(
        "python3 -c 'import torch; "
        "print(\"CUDA available:\", torch.cuda.is_available()); "
        "print(\"CUDA device count:\", torch.cuda.device_count()); "
        "print(\"CUDA device name:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\")'"
    )
))

# Completion message
node.addService(pg.Execute(
    shell="sh",
    command="echo 'CloudLab c240g5 setup complete: Ubuntu22, NVIDIA Driver 525, CUDA 11.7, PyTorch'"
))

# Output the RSpec
portal.context.printRequestRSpec()
