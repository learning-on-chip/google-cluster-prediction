all:
	@echo What?

analyze: input/distribution.csv
	./bin/analyze $<

explore: input/distribution.csv
	./bin/explore --config config/explorer.json --input $<

learn: input/distribution.csv
	./bin/learn --config config/learner.json --input $<

test:
	pytest prediction

watch:
	./bin/watch 0.0.0.0:4242

input/distribution.csv: input/distribution
	./bin/index $< $@

input/%:
	${MAKE} -C input $*

.PHONY: all analyze learn test watch
