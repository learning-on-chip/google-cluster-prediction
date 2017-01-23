all:
	@echo What?

analyze_job_events: input/job_events.sqlite3
	./bin/analyze_job_events.py

analyze_task_usage: input/task_usage_distribute
	./bin/analyze_task_usage.py $<

explore_job_events: input/job_events.sqlite3
	./bin/explore_job_events.py

learn_job_events: input/job_events.sqlite3
	./bin/learn_job_events.py

input/%:
	${MAKE} -C input $*

.PHONY: all

.PHONY: analyze_job_events explore_job_events learn_job_events

.PHONY: analyze_task_usage
