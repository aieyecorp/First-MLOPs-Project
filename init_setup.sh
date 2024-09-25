echo [$(date)]: "START"
echo [$(date)]: "creating env with python 3.8 version"
#conda create --prefix ./env python=3.10 -y
#python3.10 -m venv mlops_project
echo [$(date)]: "activating the enviroment"
#source  ./mlops_project/bin/activate
source /media/thirdeye/Data/deepstream_exp/deepstream/bin/activate
echo [$(date)]: "Installing the dev requirements"
python3.10 -m pip install -r requirements.txt
echo [$(date)]: "End"
