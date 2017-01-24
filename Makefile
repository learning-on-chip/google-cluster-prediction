all:
	@echo What?

job_events_analyze: input/job_events.sqlite3
	./bin/job_events_analyze.py $<

job_events_explore: input/job_events.sqlite3
	./bin/job_events_explore.py $<

job_events_learn: input/job_events.sqlite3
	./bin/job_events_learn.py $<

task_usage_analyze: input/task_usage_distribute
	./bin/task_usage_analyze.py $<

task_usage_histogram: input/task_usage_distribute
	./bin/task_usage_histogram.py $<

input/%:
	${MAKE} -C input $*

.PHONY: all

.PHONY: job_events_analyze job_events_explore job_events_learn

.PHONY: task_usage_analyze task_usage_histogram
