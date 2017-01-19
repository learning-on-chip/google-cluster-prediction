all:
	@echo What?

analyze: input/jobs.sqlite3
	./bin/analyze.py

explore: input/jobs.sqlite3
	./bin/explore.py

learn: input/jobs.sqlite3
	./bin/learn.py

input/*:
	${MAKE} -C input $*

.PHONY: all analyze
