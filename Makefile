analyze: input/index.csv
	./bin/analyze $<

explore: input/index.csv
	./bin/explore --input $< --config config/explorer.json

learn: input/index.csv
	./bin/learn --input $< --config config/learner.json

test:
	pytest prediction

input/index.csv: input/data input/meta.sqlite3
	./bin/index --input $< --meta input/meta.sqlite3 --output $@

input/%:
	${MAKE} -C input $*

clean:
	rm -rf input/data-*
	rm -rf output

.PHONY: all analyze clean learn test
