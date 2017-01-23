all:
	@echo What?

analyze: input/job_events.sqlite3
	./bin/analyze.py

explore: input/job_events.sqlite3
	./bin/explore.py

learn: input/job_events.sqlite3
	./bin/learn.py

input/job_events.sqlite3:
	${MAKE} -C input job_events.sqlite3

.PHONY: all analyze
