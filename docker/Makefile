IMAGE_PREFIX ?= dtrt
FINETUNE_TAG ?= $(IMAGE_PREFIX)-finetune
EDGE_TAG ?= $(IMAGE_PREFIX)-edge

.PHONY: finetune edge build

finetune:
	docker build -f Dockerfile.finetune -t $(FINETUNE_TAG) ..

edge:
	docker build -f Dockerfile.edge -t $(EDGE_TAG) ..

build: finetune edge

