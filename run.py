from optimization import run_opt
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mmm", dest = "mmm_json",help="Optimization configuration")
parser.add_argument("--bc", dest = "bc_json",help="Boundary Conditions")
parser.add_argument("--mat_lib", dest = "mat_lib_json", help="Material Library")
parser.add_argument("--machine", dest = "machine_json", help="Machine Specifications")
parser.add_argument("--directory_path", dest = "directory_path", help="directory")
parser.add_argument("--request_header_json", dest = "request_header_json", help="request_header_json")

args = parser.parse_args()

# Input json files and run topology optimization
run_opt(directory_path = args.directory_path, request_header_json = args.request_header_json, mmm_json = args.mmm_json, bc_json = args.bc_json, mat_lib_json = args.mat_lib_json, machine_json = args.machine_json)
