all:
	@echo What?

analyze: input/distribution.csv
	./bin/analyze.py $<

learn: input/distribution.csv
	./bin/learn.py --input $<

watch:
	./bin/watch.py 0.0.0.0:4242

input/distribution.csv: input/distribution
	./bin/index.py $< $@

input/%:
	${MAKE} -C input $*

.PHONY: all analyze learn watch
