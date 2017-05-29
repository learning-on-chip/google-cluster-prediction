analyze: input/index.csv
	./bin/analyze --input $<

explore: input/index.csv
	./bin/explore --input $< --config config/explore.json

learn: input/index.csv
	./bin/learn --input $< --config config/learn.json

test:
	pytest prediction

input/index.csv: input/data input/extra.sqlite3
	./bin/index --input $< --extra input/extra.sqlite3 --output $@

input/%:
	${MAKE} -C input $*

clean:
	rm -rf input/data-*
	rm -rf output

.PHONY: all analyze clean learn test
