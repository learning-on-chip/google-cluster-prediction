all:
	@echo What?

analyze: input/distribution.csv
	./bin/analyze.py $<

explore: input/distribution.csv
	./bin/explorer.py --config config/explorer.json --input $<

learn: input/distribution.csv
	./bin/learn.py --config config/learner.json --input $<

test:
	nosetests --nologcapture prediction

watch:
	./bin/watch.py 0.0.0.0:4242

input/distribution.csv: input/distribution
	./bin/index.py $< $@

input/%:
	${MAKE} -C input $*

.PHONY: all analyze learn test watch
