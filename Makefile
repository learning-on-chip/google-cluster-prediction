all:
	@echo What?

task_usage_analyze: input/distribution.csv
	./bin/task_usage_analyze.py $<

task_usage_learn: input/distribution.csv
	./bin/task_usage_learn.py --input $<

task_usage_watch:
	./bin/task_usage_watch.py 0.0.0.0:4242

input/distribution.csv: input/distribution
	./bin/task_usage_index.py $< $@

input/%:
	${MAKE} -C input $*

.PHONY: all

.PHONY: task_usage_analyze task_usage_index task_usage_learn task_usage_watch
