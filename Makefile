all:
	@echo What?

analyze_job_events: input/job_events.sqlite3
	./bin/analyze_job_events.py

explore_job_events: input/job_events.sqlite3
	./bin/explore_job_events.py

learn_job_events: input/job_events.sqlite3
	./bin/learn_job_events.py

input/job_events.sqlite3:
	${MAKE} -C input job_events.sqlite3

.PHONY: all

.PHONY: analyze_job_events explore_job_events learn_job_events
