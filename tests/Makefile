tests := $(shell find . -name 'test_*.py')
tests := $(patsubst ./%.py,%,${tests})

all: ${tests}

test_%: test_%.py
	@echo "Running $<..."
	@python $<

.PHONY: all
