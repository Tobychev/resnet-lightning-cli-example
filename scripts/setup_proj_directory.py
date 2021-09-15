import argparse as arg
import configparser as con
import os
import pathlib as p
import shutil as sh
import datetime as dt

setup_proj_args = arg.ArgumentParser()
setup_proj_args.add_argument("--config", type=str, required=True,
                      help="Configuration file for the project.")

args = setup_proj_args.parse_args()

if not os.path.exists(args.config):
  raise ValueError(f"Config file {args.config} not found!")

parser = con.ConfigParser()
parser.read(args.config,encoding="utf-8")

data = p.Path(parser["paths"]["data_dir"])
logs = p.Path(parser["paths"]["logs_dir"])
proj = p.Path(parser["paths"]["project_dir"])

if parser.has_option("paths","storage_loc"):
    store = p.Path(parser["paths"]["storage_loc"])
    data_store = store / "data"
    logs_store = store / "logs"
    if not os.path.isdir(data_store):
        print(f"Creating {data_store}")
        os.mkdir(data_store)
        data.symlink_to(data_store)
    if not os.path.isdir(logs_store):
        print(f"Creating {logs_store}")
        os.mkdir(logs_store)
        logs.symlink_to(logs_store)
else:
    if os.path.isdir(data):
        print(f"Creating {data}")
        os.mkdir(data)
    if os.path.isdir(logs):
        print(f"Creating {logs}")
        os.mkdir(logs)

if not os.path.isdir(proj / "config"):
    config_dir = proj / "config"
    config_file = os.path.basename(args.config)
    print(f"Creating {config_dir} and copying to it")
    os.mkdir(config_dir)
    sh.copy(args.config,config_dir)
    os.rename(config_dir / config_file,
              config_dir / f"PROJCONF_{dt.datetime.now().strftime('%Y-%m-%d+%H:%M:%S')}_{config_file}")

proj_config = proj / "config/root_config.ini"
proj_config_content = """[DEFAULT]
save-model-dir =
gpus = 1
seed = 352351
log-interval = 500
keep-top-N-checkpoints = 5,
terminate_on_nan = True
max-epochs = 20
batch-size = 16
loader-workers = 4
"""
with open(proj_config,"w") as fil:
    fil.writelines(proj_config_content)
