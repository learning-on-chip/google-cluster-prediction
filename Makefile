OUTPUT ?= output

all:
	@echo What?

analyze: ${OUTPUT}/jobs.sqlite3
	./bin/analyze.py

${OUTPUT}/jobs.sqlite3: data/output/job_events.sqlite3
	mkdir -p ${OUTPUT}
	cp $< $@
	cat assets/jobs.sql | sqlite3 $@

data/*:
	${MAKE} -C data $*

.PHONY: all analyze
