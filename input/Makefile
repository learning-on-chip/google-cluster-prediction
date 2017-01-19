jobs.sqlite3: data/output/job_events.sqlite3
	cp $< $@
	cat jobs.sql | sqlite3 $@
