analyze: input/distribution.csv
	./bin/analyze $<

explore: input/distribution.csv
	./bin/explore --input $< --config config/explorer.json

learn: input/distribution.csv
	./bin/learn --input $< --config config/learner.json

test:
	pytest prediction

input/distribution.csv: input/distribution input/meta.sqlite3
	./bin/index --input input/distribution --meta input/meta.sqlite3 --output $@

input/%:
	${MAKE} -C input $*

.PHONY: all analyze learn test
