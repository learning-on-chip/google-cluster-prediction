job_events.sqlite3: data/output/job_events.sqlite3
	cp $< $@
	cat job_events.sql | sqlite3 $@

task_usage_distribution: data/output/task_usage/.downloaded data/output/task_usage/distribute/.processed
	ln -sf data/output/task_usage/distribute task_usage_distribution

data/%:
	${MAKE} -C data $*
