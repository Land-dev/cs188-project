# Drone Simulation Project

need to clone gym-pybullet-drones repo 
go to https://github.com/utiasDSL/gym-pybullet-drones and clone into the gym-pybullet-drones folder

git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

You just need to get this package working and then everything else will work