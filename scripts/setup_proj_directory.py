import argparse as arg
import configparser as con
import os
import pathlib as p

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

if parser.has_option("paths","storage_loc"):
    store = p.Path(parser["paths"]["storage_loc"])
    data_store = store / "data"
    logs_store = store / "logs"
    if not os.path.isdir(data_store):
        print(f"Creating {data_store}")
        os.mkdir(data_store)
    if not os.path.isdir(logs_store):
        print(f"Creating {logs_store}")
        os.mkdir(logs_store)
    data.symlink_to(data_store)
    logs.symlink_to(logs_store)
else:
    if os.path.isdir(data):
        print(f"Creating {data}")
        os.mkdir(data)
    if os.path.isdir(logs):
        print(f"Creating {logs}")
        os.mkdir(logs)

