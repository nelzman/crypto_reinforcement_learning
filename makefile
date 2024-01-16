python_local_environment_name := local_dev
make_file_path := $(abspath $(lastword $(MAKEFILE_LIST)))
cwd := $(dir $(make_file_path))

new_environment:
	conda env create --file infrastructure/environment.yml

update_environment:
	conda env update --name crypto_reinforcement_learning --file environment.yml

export_environment:
	conda env export --name crypto_reinforcement_learning > environment.yml

remove_environment:
	conda env remove --name crypto_reinforcement_learning

start_app: 
	python3 -m src.main

build_app: 
	buildozer android debug deploy run