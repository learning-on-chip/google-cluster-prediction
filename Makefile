all:
	@echo What?

analyze: input/jobs.sqlite3
	./bin/analyze.py

input/*:
	${MAKE} -C input $*

.PHONY: all analyze
