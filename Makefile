.PHONY: help checkpoint install tests format docker
default: help

checkpoint: dataset=ImageNet
checkpoint: resolution=224x224
checkpoint: model=EfficientViT-B3

checkpoint: # Download the checkpoints
	@mkdir -p assets/checkpoints;
	@pip freeze | grep -q "^gdown" || pip install gdown;\
	wget -qO- https://raw.githubusercontent.com/mit-han-lab/efficientvit/master/README.md | grep -ozP "(?<=${dataset})[\s\S]*?\|\s*${model}\s*\|\s*${resolution}.*?\[link\]\(https:\/\/drive\.google\.com\/file\/d\/\K.*(?=\/view\?usp=(share_link|sharing)\)\s*\|)|(?<=${dataset})[\s\S]*?\|\s*${model}\s*\|\s*${resolution}.*?\[link\]\(\K.*(?=\)\s*\|)" | xargs -r0 -I {} sh -c 'echo "{}" | grep -q "^https:\/\/" && wget -o assets/checkpoints/${dataset}-${model}-${resolution}.pt {} || gdown -O assets/checkpoints/${dataset}-${model}-${resolution}.pt {}; wait'

install: # Install python dependencies
	pip install -e .

tests: # Run unit tests
	mojo run tests.mojo

docker: # Make the docker image
	docker image build -t efficentvit.mojo:latest .

format: # Format the mojo src files
	mojo format $(shell find . -type f -name '*.mojo')

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m\n\t$$(echo $$l | cut -f 2- -d'#')\n"; done
